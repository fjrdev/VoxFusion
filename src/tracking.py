import torch
import rospy
import numpy as np
import threading
from tqdm import tqdm
from criterion import Criterion
from frame import RGBDFrame
from utils.import_util import get_property
from utils.profile_util import Profiler
from variations.render_helpers import fill_in, render_rays, track_frame
import message_filters


class Tracking:
    def __init__(self, args, data_stream, logger, vis, **kwargs):
        self.args = args
        self.last_frame_id = 0
        self.last_frame = None

        self.data_stream = data_stream
        self.logger = logger
        self.visualizer = vis
        self.loss_criteria = Criterion(args)

        self.render_freq = args.debug_args["render_freq"]
        self.render_res = args.debug_args["render_res"]

        self.voxel_size = args.mapper_specs["voxel_size"]
        self.N_rays = args.tracker_specs["N_rays"]
        self.num_iterations = args.tracker_specs["num_iterations"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.learning_rate = args.tracker_specs["learning_rate"]
        self.start_frame = args.tracker_specs["start_frame"]
        self.end_frame = args.tracker_specs["end_frame"]
        self.show_imgs = args.tracker_specs["show_imgs"]
        self.step_size = args.tracker_specs["step_size"]
        self.keyframe_freq = args.tracker_specs["keyframe_freq"]
        self.max_voxel_hit = args.tracker_specs["max_voxel_hit"]
        self.max_distance = args.data_specs["max_depth"]
        self.step_size = self.step_size * self.voxel_size

        if self.end_frame <= 0:
            self.end_frame = len(self.data_stream)

        # sanity check on the lower/upper bounds
        # no need due to stream
        # self.start_frame = min(self.start_frame, len(self.data_stream))
        # self.end_frame = min(self.end_frame, len(self.data_stream))

        # profiler
        verbose = get_property(args.debug_args, "verbose", False)
        self.profiler = Profiler(verbose=verbose)
        self.profiler.enable()

    def process_first_frame(self, kf_buffer, data_in):
        # set pose to 4x4 unit matrix at beginning
        # TODO: get provided pose from ORB-SLAM2
        init_pose = np.eye(4)
        #fid, rgb, depth, K, _ = self.data_stream[self.start_frame]
        first_frame = RGBDFrame(data_in[0], data_in[1], data_in[2], data_in[3], init_pose)
        first_frame.pose.requires_grad_(False)
        first_frame.optim = torch.optim.Adam(first_frame.pose.parameters(), lr=1e-3)

        print("******* initializing first_frame:", first_frame.stamp)
        kf_buffer.put(first_frame, block=True)
        self.last_frame = first_frame
        self.start_frame += 1

    def spin(self, share_data, kf_buffer):
        print("******* tracking process started! *******")
        #import cv2
        
        #TODO: use message_filters instead of threads!
        # http://wiki.ros.org/message_filters#Time_Synchronizer
        
        rgb_b = False
        depth_b = False
        
        publish_t = threading.Thread(target=pub)
        publish_t.start()
        data_t = threading.Thread(target=listener)
        data_t.start()
        
        frame_id = 0
        rgb = np.zeros((540, 960, 3))
        depth = np.zeros((540, 960, 1))
        # hard-coded camera intrinsics
        k = np.array([805.110054612012, 427.0664954600349, 804.744879684146, 269.1892972689079])
        data_in = np.array([frame_id, rgb, depth, k])
        
        def pub():
            
            while frame_id < 1029:
                
                while not rgb_b and not depth_b:
                    pass
                
                if frame_id == 0:
                    self.process_first_frame(kf_buffer, data_in)
                    rgb_b = False
                    depth_b = False
                    continue
                rgb_b = False
                depth_b = False
                try:
                    # "*" unzips the np array
                    current_frame = RGBDFrame(*data_in)
                    self.do_tracking(share_data, current_frame, kf_buffer)

                    if self.render_freq > 0 and (frame_id + 1) % self.render_freq == 0:
                        self.render_debug_images(share_data, current_frame)
                except Exception as e:
                    print("error in dataloading: ", e, f"skipping frame {frame_id}")
                frame_id += 1
                
        
        def image(image_data):
            # convert to np-array
            image_arr = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
            data_in[1] = image_arr
            rgb_b = True
    
        def depth(depth_data):
            # convert to np-array
            disparity_arr = np.frombuffer(depth_data.image.data, dtype=np.float32).reshape(depth_data.image.height, depth_data.image.width, -1)
    
            # get depth information
            focal_length = 805.110054612012
            baseline = 4.858120401781332
            f = lambda x: (focal_length * baseline) / x
            depth_arr = f(disparity_arr)
            data_in[2] = depth_arr
            depth_b = True
        
        def listener():
            rospy.init_node('dense_map', anonymous=True)
            rospy.Subscriber("/stereo/disparity", DisparityImage, depth)
            rospy.Subscriber("/stereo/left/image_raw", Image, image)
            rospy.spin()
            

        '''
        if share_data.stop_tracking:
            break
        try:
            data_in = self.data_stream[frame_id]
    
            if self.show_imgs:
                img = data_in[1]
                depth = data_in[2]
                cv2.imshow("img", img.cpu().numpy())
                cv2.imshow("depth", depth.cpu().numpy())
                cv2.waitKey(1)

            # "*" unzips the np array
            current_frame = RGBDFrame(*data_in)
            self.do_tracking(share_data, current_frame, kf_buffer)

            if self.render_freq > 0 and (frame_id + 1) % self.render_freq == 0:
                self.render_debug_images(share_data, current_frame)
        except Exception as e:
                    print("error in dataloading: ", e, f"skipping frame {frame_id}")
        '''
        
        
        share_data.stop_mapping = True
        print("******* tracking process died *******")

    def check_keyframe(self, check_frame, kf_buffer):
        try:
            kf_buffer.put(check_frame, block=True)
        except:
            pass

    def do_tracking(self, share_data, current_frame, kf_buffer):
        decoder = share_data.decoder.cuda()
        map_states = share_data.states
        for k, v in map_states.items():
            map_states[k] = v.cuda()

        self.profiler.tick("track frame")
        frame_pose, optim, hit_mask = track_frame(
            self.last_frame.pose,
            current_frame,
            map_states,
            decoder,
            self.loss_criteria,
            self.voxel_size,
            self.N_rays,
            self.step_size,
            self.num_iterations,
            self.sdf_truncation,
            self.learning_rate,
            self.max_voxel_hit,
            self.max_distance,
            profiler=self.profiler,
            depth_variance=True
        )
        self.profiler.tok("track frame")

        current_frame.pose = frame_pose
        current_frame.optim = optim
        current_frame.hit_ratio = hit_mask.sum() / self.N_rays
        self.last_frame = current_frame

        self.profiler.tick("transport frame")
        self.check_keyframe(current_frame, kf_buffer)
        self.profiler.tok("transport frame")

        share_data.push_pose(frame_pose.translation().detach().cpu().numpy())

    @torch.no_grad()
    def render_debug_images(self, share_data, current_frame):
        rgb = current_frame.rgb
        depth = current_frame.depth
        rotation = current_frame.get_rotation()
        ind = current_frame.stamp
        w, h = self.render_res
        final_outputs = dict()

        decoder = share_data.decoder.cuda()
        map_states = share_data.states
        for k, v in map_states.items():
            map_states[k] = v.cuda()

        rays_d = current_frame.get_rays(w, h).cuda()
        rays_d = rays_d @ rotation.transpose(-1, -2)

        rays_o = current_frame.get_translation()
        rays_o = rays_o.unsqueeze(0).expand_as(rays_d)

        rays_o = rays_o.reshape(1, -1, 3).contiguous()
        rays_d = rays_d.reshape(1, -1, 3)

        final_outputs = render_rays(
            rays_o,
            rays_d,
            map_states,
            decoder,
            self.step_size,
            self.voxel_size,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            chunk_size=5000,
            return_raw=True
        )

        rdepth = fill_in((h, w, 1),
                         final_outputs["ray_mask"].view(h, w),
                         final_outputs["depth"], 0)
        rcolor = fill_in((h, w, 3),
                         final_outputs["ray_mask"].view(h, w),
                         final_outputs["color"], 0)
        # self.logger.log_raw_image(ind, rcolor, rdepth)

        # raw_surface=fill_in((h, w, 1),
        #                  final_outputs["ray_mask"].view(h, w),
        #                  final_outputs["raw"], 0)
        # self.logger.log_data(ind, raw_surface, "raw_surface")
        self.logger.log_images(ind, rgb, depth, rcolor, rdepth)

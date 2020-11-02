from processors.dp_pose_resnet_solver import DPProcessor

if __name__ == '__main__':
    ddp_processor = DPProcessor(cfg_path="configs/dp_fast_pose.yaml")
    ddp_processor.run()

from processors.dp_fastpose_processor import DPProcessor

if __name__ == '__main__':
    ddp_processor = DPProcessor(cfg_path="configs/dp_fast_pose.yaml")
    ddp_processor.run()

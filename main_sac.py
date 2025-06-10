if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_sac.yaml"
    from SAC import main
    main(config_path)

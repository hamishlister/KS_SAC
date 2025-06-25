if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="configs/config_sac.yaml")
    parser.add_argument("--ensemble_index", type=int, default=1)
    args = parser.parse_args()

    from SAC import main
    main(config_path=args.config, ens_idx=args.ensemble_index-1)

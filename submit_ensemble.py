from src import submit_ensemble
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-task Learning for TREC-IS 2020A, 2020B and 2021A.')
    parser.add_argument('--edition', type=str, default="2020b", choices=["2020a", "2020b", "2021a"],
                        help='which edition?')
    run_args = parser.parse_args()
    load_path = f"data/{run_args.edition}/subs"
    src_runs = [f"{run_args.edition}-bert-base-uncased-single",
                f"{run_args.edition}-electra-base-discriminator-single",
                f"{run_args.edition}-deberta-base-single",
                f"{run_args.edition}-distilbert-base-uncased-single"]
    submit_ensemble(src_runs, load_path, edition=run_args.edition,
                    runtag=f"{run_args.edition}_bert_electra_deberta_distilbert_ensemble")


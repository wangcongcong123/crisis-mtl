import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if you wanna use multiple gpus
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from src import Args, MTLTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-task Learning for TREC-IS 2020A, 2020B and 2021A.')
    parser.add_argument('--edition', type=str, default="2020b", choices=["2020a", "2020b", "2021a"],
                        help='which edition?')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased",
                        help='the base pre-trained transformer model for multi-task training')
    run_args = parser.parse_args()
    data_path = f"data/{run_args.edition}"
    model_short_name = run_args.model_name.split('/')[-1]
    # check the Args class for configuring more parameters
    args = Args(
        data_path=data_path,
        output_path=data_path,
        train_batch_size_per_device=32,
        override=True,
        base_model_path_or_name=run_args.model_name,
        eval_steps=-1,
        train_epochs=10,
        eval_batch_size_per_device=1024
    )
    trainer = MTLTrainer(args)
    # if a trained model available and you want to test with it directly, comment out the following line
    # and set the load_path param of trainer.predict to your trained model path
    trainer.train(eval_set="val")
    if run_args.edition == "2020a":
        # we have gts for 2020a, so we just report the eval scores without making submissions
        trainer.predict(set_name="test", with_label=True)
    else:
        # assume we do not have gts for 2020b and 2021a, so we make a submission with the predictions on the test set
        outs = trainer.predict(set_name="test")
        trainer.submit(outs, edition=run_args.edition, runtag=f"{run_args.edition}-{model_short_name}-single")

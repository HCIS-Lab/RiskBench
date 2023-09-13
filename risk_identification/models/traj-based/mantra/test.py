import argparse
import evaluate_batch


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument(
        "--model", default='pretrained_weight/model_controller_2022-09-07')
    parser.add_argument("--visualize_dataset", default=False)
    parser.add_argument("--saved_memory", default=True)
    parser.add_argument("--evaluate_or_inference", default='inference')
    parser.add_argument("--val_or_test", default='test')
    parser.add_argument("--memories_path",
                        default='pretrained_models/carla_dataset_all/')
    parser.add_argument("--preds", type=int, default=1)
    parser.add_argument(
        "--dataset_file", default="data_carla_risk_all", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of evaluation. '
                                                             'It will use for name of the test folder.')
    parser.add_argument(
        "--data_type", default="interactive", help="data_type: non-interactive, interactive, obstacle, collision")
    return parser.parse_args()


def main(config):
    v = evaluate_batch.Validator(config)
    print('start evaluation')
    v.test_model()


if __name__ == "__main__":
    config = parse_config()
    main(config)

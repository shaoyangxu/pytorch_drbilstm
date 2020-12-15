from argparse import ArgumentParser
def create_parser():
    parser = ArgumentParser()
    # model
    parser.add_argument("--multimodel", action="store_true")
    parser.add_argument("--pooling_method", default="max_avg_max_avg")
    parser.add_argument('--hidden_size', help='hidden_size', type=int, default = 450)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
    parser.add_argument("--epochs",type=int, default=100)
    parser.add_argument("--lr",type=float, default=0.0004)
    parser.add_argument('--num_layer', help='num_layer',type=int, default = 1)
    parser.add_argument("--dropout",type =float, default=0.4)
    parser.add_argument("--embedding_dropout", type=float, default=0.3)
    parser.add_argument("--num_classes",default=3)

    # data
    parser.add_argument('--checkpoint_path', help='checkpoint_path',default="train.log")
    parser.add_argument('--save_path', help='save_path', default='1234')
    parser.add_argument("--data_dir",type=str,default="SNLI0.1")
    parser.add_argument("--train_data",default="train_data.pkl")
    parser.add_argument("--valid_data",default="dev_data.pkl")
    parser.add_argument("--test_data", default="test_data.pkl")
    parser.add_argument("--embeddings",default="embeddings.pkl")
    # parser.add_argument("--train_data",default="data/train_data.pkl")
    # parser.add_argument("--valid_data",default="data/dev_data.pkl")
    # parser.add_argument("--test_data", default="data/test_data.pkl")
    # parser.add_argument("--embeddings",default="data/embeddings.pkl")
    parser.add_argument("--test_statistics",default="data/test_statistics.pkl")
    
    # train
    parser.add_argument("--optim",type = str, default="rmsprop")
    parser.add_argument("--patience",type=int, default=3)
    parser.add_argument("--max_gradient_norm",default=10.0)
    
    # other
    parser.add_argument("--local_rank",type = int,default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_from", action="store_true")
    # multimodel
    parser.add_argument("--load_path", type=str, default="trained_multi_model")
    parser.add_argument("--ensemble_mode", type=int, default=3)
    parser = parser.parse_args()
    return parser

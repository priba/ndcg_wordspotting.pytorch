import torch
import pickle
import os
from utils import cosine_similarity_matrix
from models import StringEmbedding
from datasets import build as build_dataset
from shutil import copyfile, rmtree
from sklearn.manifold import TSNE


def main(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.manual_seed(args.seed)

    train_file, _, _ = build_dataset(args.dataset, args.data_path, partition=args.partition)

    embedding_file = os.path.join(args.embedding_path, 'str_embedding.pickle')
    with open(embedding_file, 'rb') as handler:
        str_embeddings = pickle.load(handler)
    str_labels = str_embeddings[1]
    str_embeddings = str_embeddings[0]

    embedding_file = os.path.join(args.embedding_path, 'img_embedding.pickle')
    with open(embedding_file, 'rb') as handler:
        img_embeddings = pickle.load(handler)
    img_id = img_embeddings[2]
    img_labels = img_embeddings[1]
    img_embeddings = img_embeddings[0]

    ranking = cosine_similarity_matrix(str_embeddings, img_embeddings)

    word_path = os.path.join(args.data_path, args.partition, 'test')
    for label in ['great', 'recruits', 'honour', 'deliver']:
        retrieval_list = ranking[str_labels == label].argsort(dim=1, descending=True)
        retrieval_labels = img_labels[retrieval_list.cpu()]
        retrieval_id = img_id[retrieval_list.cpu()].squeeze()
        label_folder = os.path.join(args.embedding_path, label)
        if os.path.exists(label_folder):
            rmtree(label_folder)
        os.mkdir(label_folder)
        for i, file_name in enumerate(retrieval_id[:10]):

            copyfile(os.path.join(word_path, file_name), os.path.join(label_folder, f'{i}.png'))

    # OOV
    str_model = StringEmbedding(args.out_dim, train_file.voc_size()).to(device)
    checkpoint = torch.load(args.load, map_location=device)
    str_model.load_state_dict(checkpoint['str_state_dict'])
    for label in ['hous', 'capatain', 'comedian']:
        word_embedding = str_model(train_file.query_to_tensor(label).to(device).unsqueeze(0))
        retrieval_list = cosine_similarity_matrix(word_embedding, img_embeddings).argsort(descending=True)
        retrieval_labels = img_labels[retrieval_list.cpu()]
        retrieval_id = img_id[retrieval_list.cpu()].squeeze()
        label_folder = os.path.join(args.embedding_path, label)
        if os.path.exists(label_folder):
            rmtree(label_folder)
        os.mkdir(label_folder)
        for i, file_name in enumerate(retrieval_id[:10]):

            copyfile(os.path.join(word_path, file_name), os.path.join(label_folder, f'{i}.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Qualitative evaluation', add_help=True)
    parser.add_argument('--embedding_path', type=str)
    parser.add_argument('--out_dim', default=64, type=int)
    # Dataset
    parser.add_argument('--dataset', default='gw')
    parser.add_argument('--partition', default='cv1', choices=['cv1', 'cv2', 'cv3', 'cv4'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    if args.load == None:
        args.load = os.path.join(args.embedding_path, 'checkpoint_map.pth')
    main(args)


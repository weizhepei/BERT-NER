"""split the conll dataset for our model and build tags"""
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll', help="Directory containing the dataset")


def load_dataset(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset) as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                word = line.split()[0]
                tag = line.split()[-1]
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    """Write sentences.txt and tags.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print('Saving in {}...'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
        open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print('- done.')

def build_tags(data_dir, tags_file):
    """Build tags from dataset
    """
    data_types = ['train', 'val', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    with open(tags_file, 'w') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = 'data/' + args.dataset
    path_train = data_dir + '/train_bio'
    path_val = data_dir + '/val_bio'
    path_test = data_dir + '/test_bio'
    msg = f'{path_train} or {path_test} file not found. Make sure you have downloaded the right dataset'
    assert os.path.isfile(path_train) and os.path.isfile(path_test), msg
    
    # Load the dataset into memory
    print('Loading ' + args.dataset.upper() + ' dataset into memory...')
    train = load_dataset(path_train)
    test = load_dataset(path_test)
    if os.path.exists(path_val):
        val = load_dataset(path_val)
    else:
        total_train_len = len(train)
        split_val_len = int(total_train_len * 0.05)
        order = list(range(total_train_len))
        random.seed(2019)
        random.shuffle(order)

        # Split the dataset into train, val(split with shuffle) and test
        val = [train[idx] for idx in order[:split_val_len]] 
        train = [train[idx] for idx in order[split_val_len:]]
    
    save_dataset(train, data_dir + '/train')
    save_dataset(val, data_dir + '/val')
    save_dataset(test, data_dir + '/test')

    # Build tags from dataset
    build_tags(data_dir, data_dir + '/tags.txt')


import argparse

import pandas as pd
import os
import utils


class DataPreparing(object):
    def __init__(self,
                 dataset_path,
                 labels,
                 output_path,
                 create_all=False):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.train = None
        self.valid = None
        self.test = None
        self.classes = labels
        self.n_classes = 0
        self.speakers = []
        self.n_speakers = 0
        self.create_all = create_all

    def create_dataframe(self):
        print("Prepare data ...")
        valid = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        valid_lines = utils.load_txt(os.path.join(self.dataset_path, 'validation_list.txt'))
        # print(valid_lines)
        for line in valid_lines:
            parsing = line.split('/')
            vocab = parsing[0]
            if vocab in self.classes:
                file_name = line
                speaker = parsing[1].split('_')[0]
                valid["speaker"].append(speaker)
                valid["vocab"].append(vocab)
                valid["file_name"].append(file_name)

        test = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        test_lines = utils.load_txt(os.path.join(self.dataset_path, 'testing_list.txt'))
        for line in test_lines:
            parsing = line.split('/')
            vocab = parsing[0]
            file_name = line
            if vocab in self.classes:
                speaker = parsing[1].split('_')[0]
                test["speaker"].append(speaker)
                test["vocab"].append(vocab)
                test["file_name"].append(file_name)

        train = {
            "file_name": [],
            "speaker": [],
            "vocab": []
        }
        for idx, name in enumerate(self.classes):
            data_path = os.path.join(self.dataset_path, name)
            files = os.listdir(data_path)
            for i, file in enumerate(files):
                path = name + '/' + file
                # print(path)
                if file.endswith(".wav") and path not in test_lines and path not in valid_lines:
                    parsing = file.split("_")
                    speaker = parsing[0]
                    file_name = name + "/" + file
                    train["speaker"].append(speaker)
                    train["vocab"].append(name)
                    train["file_name"].append(file_name)

                    if speaker not in self.speakers:
                        self.speakers.append(speaker)

        self.n_speakers = len(self.speakers)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        df_test = pd.DataFrame(test)
        df_test.to_csv(os.path.join(self.output_path, 'test.csv'), index=False)
        df_valid = pd.DataFrame(valid)
        df_valid.to_csv(os.path.join(self.output_path, 'valid.csv'), index=False)

        df_train = pd.DataFrame(train)
        df_train.to_csv(os.path.join(self.output_path, 'train.csv'), index=False)
        print("Train: %d, Valid: %d, Test: %d" % (len(df_train), len(df_valid), len(df_test)))
        if self.create_all:
            df = pd.concat([df_train, df_valid, df_test], ignore_index=True)
            df.to_csv(os.path.join(self.output_path, 'data.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare and visualize dataset statistics")
    parser.add_argument(
        "-config_file", default="configs.yaml", help="path to config file", type=str
    )
    parser.add_argument(
        "-create_all", help="optional for create dataframe", type=bool, default=True
    )
    parser.add_argument(
        "-output_folder", help="optional for create dataframe", type=str
    )
    args = parser.parse_args()
    config_file = args.config_file
    create_all = args.create_all
    output_folder = args.output_folder

    configs = utils.load_config_file(os.path.join('./configs', config_file))
    dataset_cfgs = configs['Dataset']

    # Prepare data
    data_preparing = DataPreparing(dataset_cfgs['root_dir'],
                                   dataset_cfgs['labels'],
                                   output_folder,
                                   create_all=create_all)
    data_preparing.create_dataframe()


if __name__ == '__main__':
    main()

import librosa
import yaml


def load_config_file(file_path: str):
    """
    Load config file
    :param file_path: path to config
    :return:
    """
    try:
        yaml_config_file = open(file_path)
        file = yaml.load(yaml_config_file, Loader=yaml.FullLoader)
        return file
    except FileNotFoundError:
        print("Can not open config file.")
    return None


def load_txt(path):
    """
    Load txt file
    :param path: path to *.txt
    :return:
    """
    try:
        f = open(path, "r")
        lines = f.read().splitlines()
        return lines
    except FileNotFoundError:
        print("Can not read text file.")
    return None


def label_to_index(labels: object, label: str):
    """
    Convert label to index
    :param labels: List of labels
    :param label: Label of input/prediction
    :return: Index of label in integer
    """
    return labels.index(label)


def index_to_label(labels: object, index: int):
    """
    Convert index to label(class)
    :param labels: List of labels
    :param index: the index
    :return: the label of index in classes
    """
    return labels[index]


def load_audio(path: str, sample_rate: int):
    """
    Load audio from path with sample rate
    :param path: Path to audio
    :param sample_rate: The sample rate of audio
    :return: sample, sample_rate
    """
    sample, sample_rate = librosa.load(path=path, sr=sample_rate)
    return sample, sample_rate

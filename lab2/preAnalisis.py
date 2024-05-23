from functions import *
path = "brain_stroke.csv"

if __name__ == '__main__':
    processed_data = import_data(path)
    indicators = get_indicators(processed_data)
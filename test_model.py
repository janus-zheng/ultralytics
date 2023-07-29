''' autodocstring '''
import os
import json
import argparse

import pandas as pd
from imgcat import imgcat
from ultralytics import YOLO


def process_images(input_dir, output_csv, debug=False):
    """ _summary_ """
    print(f"Debug mode: {debug}")

    # Load the color classification model
    model = YOLO('./runs/classify/train4/weights/best.pt')  # load a pretrained model (recommended for training)

    results = model.predict(source=input_dir)
    print(f"Number of results: {len(results)}")

    result_list = []
    for res in results:
        filename = os.path.basename(res.path)
        best_color = res.names[res.probs.top1]
        top3_color = [res.names[idx] for idx in res.probs.top5[:3]]
        top3_prob = res.probs.top5conf.cpu().numpy().tolist()[:3]
        result_list.append([filename, best_color, json.dumps(top3_color), json.dumps(top3_prob)])

    df_out = pd.DataFrame(result_list, columns=["filename", "best_color", "top3_color", "top3_prob"])
    df_out.to_csv(output_csv, index=False)
    return 0


def main():
    """ main function """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='The input directory of images')
    parser.add_argument('--output-csv', type=str, required=True,
                        help='The csv file of predicted colors')
    parser.add_argument('--debug', action="store_true",
                        help='whether to print more info in debug mode')
    args = parser.parse_args()

    process_images(args.input_dir, args.output_csv, args.debug)


if __name__ == '__main__':
    main()

import glob
import json
import math
import os
import json
from os.path import expanduser
from matplotlib import pyplot as plt
import numpy as np


import delegator

dirname = 'data_named'
files_limit = 10

def get_new(x, y):
    z = np.polyfit(x, y, 6)
    f = np.poly1d(z)
    y = f(x)
    return x, y

def correct_format(pattern, limit=files_limit):
    num = 1
    for filename in sorted(glob.glob(pattern)):
        if num > limit:
            break
        command = 'sox {} -b 16 -e signed-integer {}'.format(filename, filename[:-4]+'-corrected.wav')
        print('correcting_command', command)
        res = os.system(command)
        print('correcting result', res)
        num += 1


def run_predictions(pattern, limit=files_limit):
    model_file_path = 'deepspeech-0.9.3-models.pbmm'
    scorer_file_path = 'deepspeech-0.9.3-models.scorer'
    audiorate = '16000'

    print(pattern)
    print(glob.glob(pattern))
    results = []
    num = 1
    for filename in sorted(glob.glob(pattern)):
        if num > limit:
            break
        print('filename', filename)

        wavefile = filename

        convert_command = ' '.join(
            [
                "ffmpeg",
                "-i",
                "'{}'".format(filename),
                "-ar",
                audiorate,
                "'{}'".format(wavefile),
            ]
        )
        if not os.path.isfile(wavefile):
            print(convert_command)
            r = delegator.run(convert_command)
            print(r.out)
        else:
            print('skipping wave conversion that exists')

        command = ' '.join(
            [
                "deepspeech",
                "--model",
                model_file_path,
                "--scorer",
                scorer_file_path,
                "--audio",
                "'{}'".format(wavefile),
                #            "--extended",
                "--json",
            ]
        )
        print(command)
        r = delegator.run(command)
        results.append(r.out)
        num += 1
    return results

def get_scores_from_json(results):
    print(results)
    scores = []
    for result in results:
        js = json.loads(result)
        # comment this line and uncomment next to get log of score
        score = js['transcripts'][0]['confidence']
        # scores.append(-1 * math.log(-1 * score))
        scores.append(score)
    return scores



if __name__ == '__main__':
    pattern = dirname + '/' + '*' + 'mixed.wav'
    correct_format(pattern)

    pattern = dirname + '/' + '*' + 'result.wav'
    correct_format(pattern)

    pattern = dirname + '/' + '*' + 'target.wav'
    correct_format(pattern)

    mixed_pattern = dirname + '/' + '*' + 'mixed-corrected.wav'
    mixed_results = run_predictions(mixed_pattern)

    results_pattern = dirname + '/' + '*' + 'result-corrected.wav'
    result_results = run_predictions(results_pattern)

    target_pattern = dirname + '/' + '*' + 'target-corrected.wav'
    target_results = run_predictions(target_pattern)

    mixed_conf = get_scores_from_json(mixed_results)
    result_conf = get_scores_from_json(result_results)
    target_conf = get_scores_from_json(target_results)
    print(mixed_conf, result_conf, target_conf)

    # uncomment get_new function usage to get smooth result
    y1 = mixed_conf
    x1 = [i for i in range(len(mixed_conf))]
    # x1, y1 = get_new(x1, y1)
    plt.plot(x1, y1, label='Given audio')

    y2 = result_conf
    x2 = [i for i in range(len(result_conf))]
    # x2, y2 = get_new(x2, y2)
    plt.plot(x2, y2, label='Filtered audio')

    y3 = target_conf
    x3 = [i for i in range(len(target_conf))]
    # x3, y3 = get_new(x3, y3)
    plt.plot(x3, y3, label='Target audio')

    plt.xlabel('# of audio')
    plt.ylabel('confidence')
    plt.title('DeepSpeech confidence')
    plt.legend()

    plt.rcParams['figure.figsize'] = (20,30)
    plt.savefig('res-strem.png')   # save the figure to file

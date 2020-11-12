import argparse
import sys
import csv
import sklearn
import arrow
import os

output = {
    "accuracy_values": [],
    "accuracy_value": 0.0,
    "precision_values": [],
    "precision_value": 0.0,
    "recall_values": [],
    "recall_value": 0.0,
    "f1_values": [],
    "f1_value": 0.0,
    "auc_prc_values": [],
    "auc_prc_value": 0.0,
    "training_time_seconds": []
}


def load_output():
    return output


def avg(numbers):
    if len(numbers) == 0:
        return sum(numbers)

    return sum(numbers) / len(numbers)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open('./%s/results/output_%s.csv' % (args.task, args.data_set[:-4]), "w").close()


def get_output(args, preprocessor, _output):
    prefix = 0
    prefix_all_enabled = 1

    predicted_label = list()
    ground_truth_label = list()

    if not args.cross_validation:
        result_dir_fold = \
            args.result_dir + \
            args.data_set.split(".csv")[0] + \
            "_0.csv"
    else:
        result_dir_fold = \
            './' + \
            args.task + \
            args.result_dir[1:] + \
            args.data_set.split(".csv")[0] + \
            "_%d" % preprocessor.data_structure['support']['iteration_cross_validation'] + ".csv"

    with open(result_dir_fold, 'r') as result_file_fold:
        result_reader = csv.reader(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_values"].append(
        sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_values"].append(
        sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))

    try:
        # we use the average precision at different threshold values as auc of the pr-curve
        # and not the auc-pr-curve with the trapezoidal rule / linear interpolation, because it could be too optimistic
        _output["auc_prc_values"].append(multi_class_prc_auc_score(ground_truth_label, predicted_label))
    except ValueError:
        print("Warning: Auc prc score can not be calculated ...")

    return _output


def multi_class_prc_auc_score(ground_truth_label, predicted_label, average='weighted'):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(ground_truth_label)

    ground_truth_label = label_binarizer.transform(ground_truth_label)
    predicted_label = label_binarizer.transform(predicted_label)

    return sklearn.metrics.average_precision_score(ground_truth_label, predicted_label, average=average)


def print_output(args, _output, index_fold):
    if args.cross_validation and index_fold < args.num_folds:
        llprint("\nAccuracy of fold %i: %f\n" % (index_fold, _output["accuracy_values"][index_fold]))
        llprint("Precision of fold %i: %f\n" % (index_fold, _output["precision_values"][index_fold]))
        llprint("Recall of fold %i: %f\n" % (index_fold, _output["recall_values"][index_fold]))
        llprint("F1-Score of fold %i: %f\n" % (index_fold, _output["f1_values"][index_fold]))
        llprint("Auc-prc of fold %i: %f\n" % (index_fold, _output["auc_prc_values"][index_fold]))
        llprint("Training time of fold %i: %f seconds\n\n" % (index_fold, _output["training_time_seconds"][index_fold]))
    else:
        llprint("\nAccuracy avg: %f\n" % (avg(_output["accuracy_values"])))
        llprint("Precision avg: %f\n" % (avg(_output["precision_values"])))
        llprint("Recall avg: %f\n" % (avg(_output["recall_values"])))
        llprint("F1-Score avg: %f\n" % (avg(_output["f1_values"])))
        llprint("Auc-prc avg: %f\n" % (avg(_output["auc_prc_values"])))
        llprint("Training time avg: %f seconds" % (avg(_output["training_time_seconds"])))


def get_mode(index_fold, args):
    if index_fold == -1:
        return "split-%s" % args.split_rate_test
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):
    """
    If fold < max number of folds in cross validation than use a specific value, else avg works. In addition, this holds for split.
    """

    if _mode != "split-%s" % args.split_rate_test and _mode != "avg":
        return _output[measure][_index_fold]
    else:
        return avg(_output[measure])


def write_output(args, _output, index_fold):

    with open('./%s%soutput_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4]), mode='a',
              newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')

        # if file is empty
        if os.stat('./%s%soutput_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4])).st_size == 0:
            writer.writerow(
                ["experiment", "mode", "validation", "accuracy", "precision", "recall", "f1-score", "auc-prc",
                 "training-time",
                 "time-stamp"])
        writer.writerow([
            "%s-%s" % (args.data_set[:-4], args.dnn_architecture),  # experiment
            get_mode(index_fold, args),  # mode
            "cross-validation" if args.cross_validation else "split-validation",  # validation
            get_output_value(get_mode(index_fold, args), index_fold, _output, "accuracy_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "precision_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "recall_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "f1_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "auc_prc_values", args),
            get_output_value(get_mode(index_fold, args), index_fold, _output, "training_time_seconds", args),
            arrow.now()
        ])

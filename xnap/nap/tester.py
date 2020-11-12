from __future__ import division
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
import xnap.utils as utils
import numpy as np


def test_prefix(args, preprocessor, process_instance, prefix_size):
    """
    Perform test for LRP.

    Select model with the highest f1-score:
    - Bi-LSTM model #9 for bpi2019s
    - Bi-LSTM model #8 for helpdesk

    :param args:
    :param preprocessor:
    :param process_instance:
    :param prefix_size:
    :return: parameters for LRP
    """

    # select the best model of the ten-fold cross-validation
    model_index = 8
    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4], model_index))

    cropped_process_instance = preprocessor.get_cropped_instance(prefix_size, process_instance)
    cropped_process_instance_label = preprocessor.get_cropped_instance_label(prefix_size, process_instance)
    test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)

    y = model.predict(test_data)
    y = y[0][:]

    prediction = preprocessor.get_event_type_max_prob(y)
    prediction_class = np.argmax(y)

    prob_dist = dict()
    for index, prob in enumerate(y):
        prob_dist[preprocessor.get_event_type(index)] = y[index]

    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])
    cropped_process_instance_label_class = preprocessor.data_structure['support']['map_event_type_to_event_id'][cropped_process_instance_label]



    return prediction_class, prediction, cropped_process_instance_label_class, cropped_process_instance_label, cropped_process_instance, model, test_data_reshaped, prob_dist


def test(args, preprocessor):
    """
    Perform test for model validation.
    :param args:
    :param preprocessor:
    :return: none
    """

    preprocessor.get_instances_of_fold('test')
    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4],
                    preprocessor.data_structure['support']['iteration_cross_validation']))

    prediction_size = 1
    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = './' + args.task + args.result_dir[1:] + data_set_name
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.data_structure['support']['iteration_cross_validation'], ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted"])

        # for prefix_size >= 2
        for prefix_size in range(2, preprocessor.data_structure['meta']['max_length_process_instance']):
            utils.llprint("Prefix size: %d\n" % prefix_size)

            for process_instance, event_id in zip(preprocessor.data_structure['data']['test']['process_instances'],
                                                  preprocessor.data_structure['data']['test']['event_ids']):

                cropped_process_instance = preprocessor.get_cropped_instance(
                    prefix_size,
                    process_instance)

                if preprocessor.data_structure['support']['end_process_instance'] in cropped_process_instance:
                    continue

                ground_truth = ''.join(process_instance[prefix_size:prefix_size + prediction_size])
                prediction = ''

                for i in range(prediction_size):

                    if len(ground_truth) <= i:
                        continue

                    test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)

                    y = model.predict(test_data)
                    y_char = y[0][:]

                    predicted_event = preprocessor.get_event_type_max_prob(y_char)

                    cropped_process_instance += predicted_event
                    prediction += predicted_event

                    if predicted_event == preprocessor.data_structure['support']['end_process_instance']:
                        print('! predicted, end of process instance ... \n')
                        break

                output = []
                if len(ground_truth) > 0:
                    output.append(event_id)
                    output.append(prefix_size)
                    output.append(str(ground_truth).encode("utf-8"))
                    output.append(str(prediction).encode("utf-8"))
                    result_writer.writerow(output)

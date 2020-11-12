from __future__ import division
import csv
import numpy
import copy
import xnap.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit


class Preprocessor(object):

    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "",
            'ascii_offset': 161,
            'data_format': "%d.%m.%Y-%H:%M:%S",
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'event_labels': [],
            'event_types': [],
            'map_event_label_to_event_id': [],
            'map_event_id_to_event_label': [],
            'map_event_type_to_event_id': [],
            'map_event_id_to_event_type': [],
            'end_process_instance': '!',
        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_process_instances': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],

            'train': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'event_ids': []
            }
        }
    }


    def __init__(self, args):

        utils.llprint("Initialization ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set
        self.get_sequences_from_eventlog()
        self.data_structure['support']['elements_per_fold'] = \
            int(round(
                self.data_structure['meta']['num_process_instances'] / self.data_structure['support']['num_folds']))


        # add end marker of process instance
        self.data_structure['data']['process_instances'] = list(map(lambda x: x + ['!'], self.data_structure['data']['process_instances']))
        self.data_structure['meta']['max_length_process_instance'] = max(map(lambda x: len(x), self.data_structure['data']['process_instances']))

        # structures for predicting next activities
        self.data_structure['support']['event_labels'] = list(
            map(lambda x: set(x), self.data_structure['data']['process_instances']))
        self.data_structure['support']['event_labels'] = list(
            set().union(*self.data_structure['support']['event_labels']))
        self.data_structure['support']['event_labels'].sort()
        self.data_structure['support']['event_types'] = copy.copy(self.data_structure['support']['event_labels'])
        self.data_structure['support']['map_event_label_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_id_to_event_label'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_type_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['support']['map_event_id_to_event_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])

        self.data_structure['meta']['num_features'] = len(self.data_structure['support']['event_labels'])

        args.dim = len(self.data_structure['support']['event_labels'])

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)


    def get_sequences_from_eventlog(self):
        """
        Get sequences form event log.
        """

        id_latest_process_instance = ''
        process_instance = []
        first_event_of_process_instance = True
        output = True

        file = open(self.data_structure['support']['data_dir'], 'r')
        reader = csv.reader(file, delimiter=';', quotechar='|')
        next(reader, None)

        for event in reader:

            id_current_process_instance = event[0]

            if output:
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')
                id_latest_process_instance = id_current_process_instance

                if not first_event_of_process_instance:
                    self.add_data_to_data_structure(process_instance, 'process_instances')

                process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            process_instance.append(event[1])
            first_event_of_process_instance = False

        file.close()

        self.add_data_to_data_structure(process_instance, 'process_instances')

        self.data_structure['meta']['num_process_instances'] += 1


    def set_training_set(self):
        """
        Set training set
        """

        utils.llprint("Get training instances ... \n")
        process_instances_train, _ = \
            self.get_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")
        cropped_process_instances, next_events = \
            self.get_cropped_instances(process_instances_train)

        utils.llprint("Create training set data as 3d-tensor ... \n")
        features_data = self.get_data_tensor(cropped_process_instances,
                                             'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_matrix(cropped_process_instances,
                                       next_events)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels


    def get_event_type_max_prob(self, predictions):
        """
        Get most likely activity from a probability distribution.
        :param predictions:
        :return: activity.
        """

        max_prediction = 0
        event_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                event_type = self.data_structure['support']['map_event_id_to_event_type'][index]
            index += 1

        return event_type


    def get_event_type(self, index):
        """
        Get activity label for activity id.
        :param index:
        :return: activity.
        """

        return self.data_structure['support']['map_event_id_to_event_type'][index]


    def add_data_to_data_structure(self, values, structure):
        """
        Add data to general data structure.
        :param values:
        :param structure:
        """

        self.data_structure['data'][structure].append(values)


    def set_indices_k_fold_validation(self):
        """
        Performs k fold cross-validation.
        """

        kFold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=0, shuffle=False)

        for train_indices, test_indices in kFold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)



    def set_indices_split_validation(self, args):
        """
        Produces indices for split-validation.
        :param args:
        """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)


    def get_instances_of_fold(self, mode):
        """
        Retrieves process instances of a fold.
        :param mode:
        :return: process instances.
        """

        process_instances_of_fold = []
        event_ids_of_fold = []

        for index, value in enumerate(self.data_structure['support'][mode + '_index_per_fold'][
                                          self.data_structure['support']['iteration_cross_validation']]):
            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            event_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])


        if mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['event_ids'] = event_ids_of_fold
            return

        return process_instances_of_fold, event_ids_of_fold


    def get_cropped_instances(self, process_instances):
        """
        Crops prefixes out of process instances.
        :param process_instances:
        :return: cropped process instances, next events
        """

        cropped_process_instances = []
        next_events = []


        for process_instance in process_instances:
            for i in range(0, len(process_instance)):

                if i == 0:
                    continue
                cropped_process_instances.append(process_instance[0:i])
                # label
                next_events.append(process_instance[i])

        return cropped_process_instances, next_events


    def get_cropped_instance_label(self, prefix_size, process_instance):
        """
        Crops the next activity label out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return: Next activity label
        """

        if prefix_size == len(process_instance) - 1:
            # end marker
            return self.data_structure["support"]["end_process_instance"]
        else:
            # label of next act
            return process_instance[prefix_size]


    def get_cropped_instance(self, prefix_size, process_instance):
        """
        Crops prefixes out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return: prefixes.
        """

        return process_instance[:prefix_size]


    def get_data_tensor(self, cropped_process_instances, mode):
        """
        Get three-order data tensor from process instances.
        :param cropped_process_instances:
        :param mode:
        :return: data tensor
        """

        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float64)
        else:
            data_set = numpy.zeros((
                1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float32)


        for index, cropped_process_instance in enumerate(cropped_process_instances):
            for index_, activity in enumerate(cropped_process_instance):

                data_set[index, index_, self.data_structure['support']['map_event_label_to_event_id'][activity]] = 1

        return data_set


    def get_data_tensor_for_single_prediction(self, cropped_process_instance):
        """
        Get three-order data tensor from a single prefix of a process instances.
        The prefix represents a running process instance.
        :param cropped_process_instance:
        :return: data tensor
        """

        data_set = self.get_data_tensor(
            [cropped_process_instance],
            'test')

        return data_set


    def get_label_matrix(self, cropped_process_instances, next_events):
        """
        Get matrix from process instances.
        :param next_events:
        :param cropped_process_instances:
        :return: label matrix
        """

        label = numpy.zeros((len(cropped_process_instances), len(self.data_structure['support']['event_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for event_type in self.data_structure['support']['event_types']:

                if event_type == next_events[index]:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 1
                else:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 0

        return label


    def get_random_process_instance(self, lower_bound, upper_bound):
        """
        Selects a random process instance from the complete event log.
        :param lower_bound:
        :param upper_bound:
        :return: process instance.
        """

        process_instances = self.data_structure['data']['process_instances']

        while True:
            rand = numpy.random.randint(len(process_instances))
            size = len(process_instances[rand])

            if lower_bound <= size <= upper_bound:
                break

        return process_instances[rand]


















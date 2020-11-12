import xnap.config as config
import xnap.utils as utils
from xnap.explanation.LSTM.LSTM_bidi import *
from xnap.explanation.util.heatmap import html_heatmap
import xnap.explanation.util.browser as browser
from xnap.nap.preprocessor import Preprocessor as Preprocessor
import xnap.nap.tester as test
import xnap.nap.trainer as train
import numpy as np

if __name__ == '__main__':
    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)

    # explanation mode for nap
    if args.explain:

        preprocessor = Preprocessor(args)

        process_instance = preprocessor.get_random_process_instance(args.rand_lower_bound, args.rand_upper_bound)
        prefix_heatmaps: str = ""

        for prefix_index in range(2, len(process_instance)):

            # next activity prediction
            predicted_act_class, predicted_act_class_str, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(args, preprocessor, process_instance, prefix_index)
            print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (prefix_index, predicted_act_class, target_act_class_str))
            print("Probability Distribution:")
            print(prob_dist)

            # compute lrp relevances
            eps = 0.001  # small positive number
            bias_factor = 0.0  # recommended value
            net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model

            Rx, Rx_rev, R_rest = net.lrp(prefix_words, predicted_act_class, eps, bias_factor)  # perform LRP
            R_words = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances
            scores = net.s.copy()  # classification prediction scores

            prefix_heatmaps = prefix_heatmaps + html_heatmap(prefix_words, R_words) + "<br>"  # create heatmap
            browser.display_html(prefix_heatmaps)  # display heatmap


    # validation mode for nap
    elif not args.explain:

        preprocessor = Preprocessor(args)

        if args.cross_validation:
            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)

        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    else:
        print("No mode selected ...")

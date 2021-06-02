# coding: utf-8

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import copy
from data_process import NeuralDataset
from utils.torch_utils import calc_f1
from utils.file_utils import from_project_root


def evaluate(model, data_url):
    """ evaluating end2end model on dataurl

    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model

    Returns:
        ret: dict of precision, recall, and f1

    """
    print("\nevaluating model on:", data_url, "\n")
    cased = True
    if "genia" not in data_url:
        cased = False
    dataset = NeuralDataset(data_url, next(model.parameters()).device, evaluating=True, cased = cased)
    loader = DataLoader(dataset, batch_size=100, collate_fn=dataset.collate_func)

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        span_sentence_true_list, span_sentence_pred_list = list(), list()
        region_true_list, region_pred_list = list(), list()

        region_length_true_list, region_length_pred_list = list(), list()
        analysis_length = 5
        region_length_true_count_list, region_length_pred_count_list = list(), list()
        for i in range(analysis_length):
            region_length_pred_count_list.append(0)
            region_length_true_count_list.append(0)
            region_length_pred_list.append(list())
            region_length_true_list.append(list())

        region_true_count, region_pred_count = 0, 0
        for data, single_region_labels, span_sentence_labels, span_region_labels, records_list in loader:
            try:
                single_region_outputs, span_sentence_outputs, span_region_outputs = model.forward(*data)
                # single_region_outputs (batch_size, n_class, max_sent_len)
                # span_sentence_outputs (batch_size, 2, lengths[0])
                # span_region_outputs (tensor_list)
                masks = data[-1]
                
                pred_span_sentence_labels = torch.argmax(span_sentence_outputs, dim=1) * masks
                # pred_span_sentence_labels (batch_size, max_len)
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                continue
            pred_single_region_labels = torch.argmax(single_region_outputs, dim=1) * masks
            # pred_single_region_labels (batch_size, max_sent_len)
            pred_span_region_labels = None
            if span_region_outputs is not None:
                pred_span_region_labels = torch.argmax(span_region_outputs, dim=1).view(-1).cpu()
            # pred_span_region_labels (list)
            lengths = data[1]
            ind = 0
            for span_sent_labels, sent_single_region_labels, length, true_records in zip(pred_span_sentence_labels,
                                                                                    pred_single_region_labels, lengths, records_list):
                pred_records = dict()

                for start in range(0, length):
                    if sent_single_region_labels[start] > 0:
                        pred_records[(start,start+1)] = sent_single_region_labels[start].item()

                    if pred_span_region_labels == None:
                        continue
                    if span_sent_labels[start]:
                        for end in range(start + 1, length):
                            if pred_span_region_labels[ind]:
                                pred_records[(start,end+1)] = pred_span_region_labels[ind].item()
                            ind += 1

                for region in true_records:
                    true_label = dataset.label_dict[true_records[region]]
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    # if torch.is_tensor(pred_label):
                    #     pred_label = pred_label.item()
                    region_pred_list.append(pred_label)
                    ## for analysis of length
                    if (region[1] - region[0]) <= analysis_length:
                        region_length_true_list[region[1] - region[0] - 1].append(true_label)
                        region_length_pred_list[region[1] - region[0] - 1].append(pred_label)
                        region_length_true_count_list[region[1] - region[0] - 1] += 1
                        if pred_label > 0:
                            region_length_pred_count_list[region[1] - region[0] - 1] += 1

                for region in pred_records:
                    if region not in true_records:
                        pred_label = pred_records[region]
                        # if torch.is_tensor(pred_label):
                        #     pred_label = pred_label.item()
                        region_pred_list.append(pred_label)
                        region_true_list.append(0)
                        ## for analysis of length
                        if (region[1] - region[0]) <= analysis_length:
                            region_length_true_list[region[1] - region[0] - 1].append(0)
                            region_length_pred_list[region[1] - region[0] - 1].append(pred_label)
                            region_length_pred_count_list[region[1] - region[0] - 1] += 1

            single_region_labels = single_region_labels.view(-1).cpu()
            span_region_labels = span_region_labels.view(-1).cpu()
            region_true_count += int((single_region_labels > 0).sum())
            region_true_count += int((span_region_labels > 0).sum())

            pred_single_region_labels = pred_single_region_labels.view(-1).cpu()
            if pred_span_region_labels is not None:
                pred_span_region_labels = pred_span_region_labels.view(-1).cpu()
                region_pred_count += int((pred_span_region_labels > 0).sum())
            region_pred_count += int((pred_single_region_labels > 0).sum())


            pred_span_sentence_labels = pred_span_sentence_labels.view(-1).cpu()
            span_sentence_labels = span_sentence_labels.view(-1).cpu()
            for tv, pv, in zip(span_sentence_labels, pred_span_sentence_labels):
                    span_sentence_true_list.append(tv)
                    span_sentence_pred_list.append(pv)

        print("===========length analyse start=============")
        for i in range(analysis_length):
            tp = 0
            for pv, tv in zip(region_length_pred_list[i], region_length_true_list[i]):
                if pv == tv == 0:
                    continue
                if pv == tv:
                    tp += 1
            fp = region_length_pred_count_list[i] - tp
            fn = region_length_true_count_list[i] - tp
            p, r, f1 = calc_f1(tp, fp, fn, print_result=False)
            print("length: %d: precision = %f, recall = %f, micro_f1 = %f\n" % (i + 1, p, r, f1))

        print("===========length analyse end=============")


        print("sentence head and tail labeling result:")
        print(classification_report(span_sentence_true_list, span_sentence_pred_list, target_names=['out-entity', 'head-entity'], digits=6))

        print("region classification result:")
        print(classification_report(region_true_list, region_pred_list, target_names=list(dataset.label_dict)[:12], digits=6))

        ret = {'precision': 0, 'recall': 0, 'f1': 0}
        tp = 0
        for pv, tv in zip(region_pred_list, region_true_list):
            if pv == tv == 0:
                continue
            if pv == tv:
                tp += 1
        fp = region_pred_count - tp
        fn = region_true_count - tp
        print(region_pred_count)
        print(region_true_count)
        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)

        return ret


def main():
    model_url = from_project_root("data/model/end2end_model_epoch1_0.743877")
    test_url = from_project_root("data/Germ/germ.test.iob2")
    model = torch.load(model_url)
    evaluate(model, test_url)
    pass


if __name__ == '__main__':
    main()

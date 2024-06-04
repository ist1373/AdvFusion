
# def calculate_f1(tokenizer,predicted_path, groundtruth_path):
#     outputs = ""
#     with open(predicted_path, 'r') as f:
#       for line in f:
#         a = line.split('\t')[1].split('\n')[0]
#         outputs += " " + a
    
#     golds = ""
#     with open(groundtruth_path, 'r') as f:
#       for line in f:
#         a = line.split('\t')[1].split('\n')[0]
#         golds += " " + a
    
#     output_tokens = tokenizer.tokenize(outputs)
#     gold_tokens = tokenizer.tokenize(golds)

#     output_set = set(output_tokens)
#     gold_set = set(gold_tokens)

#     precision = len(gold_set.intersection(output_set)) / len(output_set)
#     recall = len(gold_set.intersection(output_set)) / len(gold_set)
#     f1_score = (2 * precision * recall) / (precision + recall)

#     return {'precision':precision, 'recall':recall, 'f1_score':f1_score}


def calculate_f1(tokenizer,predicted_path, groundtruth_path):
    outputs = []
    with open(predicted_path, 'r') as f:
      for line in f:
        a = line.split('\t')[1].split('\n')[0]
        outputs.append(a)

    golds = []
    with open(groundtruth_path, 'r') as f:
      for line in f:
        a = line.split('\t')[1].split('\n')[0]
        golds.append(a)
    precisions = []
    recalls = []
    f1_scores = []
    for pred,ref in zip(outputs,golds):

      output_tokens = tokenizer.tokenize(pred)
      gold_tokens = tokenizer.tokenize(ref)
      output_set = set(output_tokens)
      gold_set = set(gold_tokens)
      if len(output_set) !=0:
        precisions.append(len(gold_set.intersection(output_set)) / len(output_set))
      else:
        precisions.append(0)
      if len(gold_set) != 0:
        recalls.append(len(gold_set.intersection(output_set)) / len(gold_set))
      else:
        recalls.append(0)

      if precisions[-1] + recalls[-1] != 0:
        f1_scores.append((2 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))
      else:
        f1_scores.append(0)
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)
    f1_score = sum(f1_scores)/len(f1_scores)

    return {'precision':precision, 'recall':recall, 'f1_score':f1_score}
  
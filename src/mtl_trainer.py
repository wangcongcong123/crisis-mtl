from transformers import AutoTokenizer, get_constant_schedule, get_constant_schedule_with_warmup, AutoModel, AdamW, \
    get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from collections import Counter
from src.args import Args
from src.eda import eda
from src.utils import *
import transformers
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import nn
from copy import deepcopy
import math
import preprocessor as pre

pre.set_options(pre.OPT.URL, pre.OPT.EMOJI)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def map_pri(score):
    if 0.75 < score <= 1.0:
        return "Critical"
    if 0.5 < score <= 0.75:
        return "High"
    if 0.25 < score <= 0.5:
        return "Medium"
    return "Low"


class MTLModelForSequenceClassification(nn.Module):
    def __init__(self, base_model_path_or_name, num_label):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_path_or_name, return_dict=True)
        self.classification_head = nn.Linear(self.base_model.config.hidden_size, num_label)
        self.regression_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, inputs):
        outputs = self.base_model(**inputs)
        classification_logits = self.classification_head(
            outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :])
        regression_logits = self.regression_head(
            outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :])
        return classification_logits, regression_logits

    def save_pretrained(self, save_path):
        self.base_model.save_pretrained(save_path)
        torch.save(self.classification_head.state_dict(), os.path.join(save_path, "classification_head.bin"))
        torch.save(self.regression_head.state_dict(), os.path.join(save_path, "regression_head.bin"))

    @classmethod
    def from_pretrained(cls, load_path, num_label):
        # todo: num_label should be saved as config file in load_path
        model = cls(load_path, num_label)
        model.classification_head.load_state_dict(
            torch.load(os.path.join(load_path, "classification_head.bin"), map_location=device))
        model.regression_head.load_state_dict(
            torch.load(os.path.join(load_path, "regression_head.bin"), map_location=device))
        return model


class MTLTrainer:
    def __init__(self, args: Args):
        transformers.logging.set_verbosity_info()
        self.logger = transformers.logging.get_logger()
        self.args = args
        self.cate_classes = read_class_names(os.path.join(self.args.data_path, "classes_categories.txt"))
        self.pri_map = priority2numeric
        add_filehandler_for_logger(self.args.data_path, self.logger)
        self.logger.info("General Args: " + json.dumps(self.args.__dict__, indent=2))
        set_seed(self.args.seed)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device_count = 1
        if self.device != "cpu":
            self.device_count = torch.cuda.device_count()

    def encode_data(self, tokenizer, examples, with_label=True):
        encoded_examples = {}
        for i in trange(0, len(examples), self.args.tok_bs):
            batch_examples = examples[i:i + self.args.tok_bs]
            batch_texts = [be["text"] for be in batch_examples]
            inputs = tokenizer(batch_texts, truncation=True, max_length=self.args.max_seq_length, return_tensors='pt',
                               padding="max_length")
            inputs.update({"raw_text": [be["text"] for be in batch_examples]})

            if with_label:
                categories = []
                for be in batch_examples:
                    categories.append(get_short_cates(be["categories"]))
                inputs.update({"categories": categories})
                inputs.update({"priority": [be["priority"] for be in batch_examples]})
                one_hots = []
                for cates in categories:
                    one_hot = [0] * len(self.cate_classes)
                    for cat in cates.split(","):
                        one_hot[self.cate_classes.index(cat)] = 1
                    one_hots.append(one_hot)

                inputs.update({"categories_indices": torch.tensor(one_hots)})
                inputs.update({"priority_score": [self.pri_map[be["priority"]] for be in batch_examples]})
                inputs.update(
                    {"events": [be["eventID"] if "eventID" in be else be["event_id"] for be in batch_examples]})

            for k, v in inputs.items():
                if k not in encoded_examples:
                    encoded_examples[k] = v
                else:
                    if torch.is_tensor(v):
                        encoded_examples[k] = torch.cat([encoded_examples[k], v], dim=0)
                    else:
                        encoded_examples[k].extend(v)

        return encoded_examples

    def get_model_by_device(self, model):
        if self.device != "cpu":
            model.to(self.device)
            model = DataParallel(model, device_ids=[i for i in range(self.device_count)])
            return model
        return model

    def aug_with_eda(self, sequence, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, alpha_rd=0.2, num_aug=5):
        aug_sentences = eda(sequence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        return aug_sentences

    def aug_batch_with_eda(self, examples, class_dist, aug_target=500):
        self.logger.info(f"start EDA to ensure each class has at least {aug_target} examples")
        aug_labels2num_aug = {}
        for key, value in class_dist.items():
            if value <= aug_target:
                aug_labels2num_aug[key] = math.ceil((aug_target - value) / value)
        aug_examples = []
        label_key = "categories"
        for example in examples:
            if get_short_cates(example[
                                   label_key]) in aug_labels2num_aug:  # this makes it stricter for multi-label dataset augmentation
                # num_aug_list = [aug_labels2num_aug[j] for j in example[label_key].split(",") if j in aug_labels2num_aug]
                # num_aug = int(sum(num_aug_list)/len(num_aug_list))
                text_ = deepcopy(example["text"])
                aug_sentences = self.aug_with_eda(pre.clean(text_), alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2,
                                                  alpha_rd=0.2,
                                                  num_aug=aug_labels2num_aug[get_short_cates(example[label_key])])
                for each in aug_sentences:
                    # example["text_"] = text_
                    new_example = deepcopy(example)
                    new_example.update({"text": each})
                    aug_examples.append(new_example)
        examples.extend(aug_examples)
        preds = []
        for i in examples:
            preds.extend(get_short_cates(i[label_key]).split(","))
        return examples, Counter(preds)

    def train(self, eval_set="val", train_set_name="train"):
        model_short_name = self.args.base_model_path_or_name.split('/')[-1]
        load_path = os.path.join(self.args.data_path, f"{model_short_name}-ft-{train_set_name}-data.pt")
        tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path_or_name)

        if os.path.isfile(load_path) and not self.args.override:
            encoded_train_examples = torch.load(load_path)
        else:
            data_path = os.path.join(self.args.data_path, f"{train_set_name}.json")
            examples = read_jsonl(data_path)
            if self.args.eda_aug:
                self.logger.info('start eda augmentation')
                categories = []
                for ex in examples:
                    cates = []
                    for each in ex["categories"].split(","):
                        cates.append(each.split("-")[-1])
                    categories.extend(cates)

                class_dist = Counter(categories)
                self.logger.info(f'dist of categories before augmentation: {json.dumps(class_dist, indent=2)}')
                examples, class_dist = self.aug_batch_with_eda(examples, class_dist, aug_target=500)
                self.logger.info(f'dist of categories after eda augmentation: {json.dumps(class_dist, indent=2)}')

            self.report_data_stats(examples)
            encoded_train_examples = self.encode_data(tokenizer, examples)
            torch.save(encoded_train_examples, load_path)

        categories = []
        for each in encoded_train_examples["categories"]:
            categories.extend(each.split(","))

        self.logger.info(f"the dist of categories (training): {json.dumps(Counter(categories), indent=2)}")
        self.logger.info(
            f"the dist of priority (training): {json.dumps(Counter(encoded_train_examples['priority']), indent=2)}")
        self.logger.info(
            f"the dist of tweets by events (training): {json.dumps(Counter(encoded_train_examples['events']), indent=2)}")

        eval_dataset = None
        if eval_set is not None and os.path.isfile(os.path.join(self.args.data_path, f"{eval_set}.json")):
            data_path = os.path.join(self.args.data_path, f"{eval_set}.json")
            examples = read_jsonl(data_path)
            encoded_eval_examples = self.encode_data(tokenizer, examples)
            eval_dataset = MyDataset(encoded_eval_examples)
            self.logger.info(
                f"the dist of tweets by events (eval): {json.dumps(Counter(encoded_eval_examples['events']), indent=2)}")

        model = MTLModelForSequenceClassification(self.args.base_model_path_or_name, len(self.cate_classes))
        train_dataset = MyDataset(encoded_train_examples)
        model = self.get_model_by_device(model)
        train_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size_per_device * self.device_count,
                                  num_workers=1, shuffle=True)
        total_steps = len(train_loader) * self.args.train_epochs / self.args.accumulation_steps

        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.args.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        optimizer = AdamW(optim_groups, lr=self.args.training_lr, eps=1e-8)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.training_args.pre_train_training_lr, eps=1e-8)

        if self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.args.warmup_ratio * total_steps,
                                                        num_training_steps=total_steps)
        elif self.args.lr_scheduler == "linearconstant":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=total_steps)
        else:
            scheduler = get_constant_schedule(optimizer)

        multi_label_loss_fn = nn.BCEWithLogitsLoss()
        regression_loss_fn = nn.MSELoss()
        model.train()

        global_step = 0
        eval_loss = 0

        for i in range(self.args.train_epochs):
            self.logger.info(f"Epoch {i + 1}:")
            wrap_dataset_loader = tqdm(train_loader)
            model.zero_grad()
            total_epoch_loss = 0
            for j, batch in enumerate(wrap_dataset_loader):
                batch.pop("categories")
                batch.pop("priority")
                batch.pop("raw_text")
                batch.pop("events")

                categories_indices = batch.pop("categories_indices").to(self.device)
                priority_score = batch.pop("priority_score").to(self.device)
                inputs = {k: batch[k].to(self.device) for k in batch}

                classification_logits, regression_logits = model(inputs)
                classification_loss = multi_label_loss_fn(classification_logits, categories_indices.float())
                regression_loss = regression_loss_fn(regression_logits.view(-1).sigmoid(), priority_score.float())
                loss = self.args.alpha * classification_loss + (1 - self.args.alpha) * regression_loss
                total_epoch_loss += loss.item()
                eval_loss += loss.item()
                loss.backward()
                if (j + 1) % self.args.accumulation_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                global_step += 1
                wrap_dataset_loader.update(1)
                wrap_dataset_loader.set_description(
                    f"MTL-Training - epoch {i + 1}/{self.args.train_epochs} iter {j}/{len(wrap_dataset_loader)}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")
                if self.args.eval_steps > 0 and global_step % self.args.eval_steps == 0:
                    self.logger.info(
                        f"\naverage training loss at global_step={global_step}: {eval_loss / self.args.eval_steps}")
                    eval_loss = 0
                    if eval_dataset is not None:
                        self.logger.info(
                            f"evaluation during training on {eval_set} set ({model_short_name}_epoch{i + 1}): ")
                        self.inference(model, eval_dataset)
                    model.train()

            self.logger.info(f"Average training loss for epoch {i + 1}: {total_epoch_loss / len(train_loader)}")
            # evaluate at the end of epoch if eval_steps is smaller than or equal to 0
            if self.args.eval_steps <= 0:
                self.logger.info(f"evaluation during training on {eval_set} set ({model_short_name}_epoch{i + 1}): ")
                self.inference(model, eval_dataset)
                model.train()
            # save up at end of each epoch!
            # model.save_pretrained(os.path.join(self.args.output_path, "mtl_train", model_short_name, f"epoch_{i + 1}"))
            # tokenizer.save_pretrained(os.path.join(self.args.output_path, "mtl_train", model_short_name, f"epoch_{i + 1}"))
        # save up at end of training!
        save_model_path = os.path.join(self.args.output_path, "mtl_train",
                                       model_short_name if not self.args.eda_aug else model_short_name + "-eda",
                                       "final_model")
        if isinstance(model, DataParallel):
            model.module.save_pretrained(save_model_path)
        else:
            model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)

        # eval at the final model saved ck
        return_dict = {}
        if eval_dataset is not None:
            self.logger.info(f"evaluation on test set with mtl-trained model: {save_model_path}")
            return_dict = {f"mtl_train(eval_set={eval_set})": self.inference(model, eval_dataset)}
        return return_dict

    def predict(self, load_path=None, set_name="test", with_label=False, data_load_path=None):
        # load up
        model_short_name = self.args.base_model_path_or_name.split('/')[-1]
        load_path = load_path if load_path is not None else os.path.join(self.args.output_path, "mtl_train",
                                                                         model_short_name if not self.args.eda_aug else model_short_name + "-eda",
                                                                         "final_model")

        self.logger.info(
            f"*************** start making predictions for {self.args.data_path + '/' if data_load_path is None else data_load_path + '/'}{set_name}.json with label = {with_label} using model: {load_path}****************")
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model = MTLModelForSequenceClassification.from_pretrained(load_path, len(self.cate_classes))

        data_path = os.path.join(self.args.data_path if data_load_path is None else data_load_path, f"{set_name}.json")
        examples = read_jsonl(data_path)  ######[:100]

        encoded_data_load_path = os.path.join(self.args.data_path if data_load_path is None else data_load_path,
                                              f"{model_short_name}-predict-{set_name}-data.pt")
        if os.path.isfile(encoded_data_load_path) and not self.args.override:
            self.logger.info(f"load encoded data for prediction from: {encoded_data_load_path}")
            encoded_examples = torch.load(encoded_data_load_path)
        else:
            self.logger.info(f"start encoding data for prediction")
            encoded_examples = self.encode_data(tokenizer, examples, with_label=with_label)
            torch.save(encoded_examples, encoded_data_load_path)
            self.logger.info(f"save encoded data for prediction to: {encoded_data_load_path}")

        predict_dataset = MyDataset(encoded_examples)
        model = self.get_model_by_device(model)
        if with_label:
            # if with label, we report the performance
            scores_dict = self.inference(model.to(self.device), predict_dataset)
            self.logger.info(json.dumps(scores_dict, indent=2))
            return scores_dict
        else:
            # if without label, we get the returned preds and use them to make a submission
            cate_outs, cate_preds, pri_outs, pri_preds = self.inference(model.to(self.device), predict_dataset,
                                                                        return_preds=True)
            assert len(examples) == len(cate_outs) == len(cate_preds) == len(pri_outs) == len(pri_preds)
            return {"examples": examples, "cate_outs": cate_outs, "cate_preds": cate_preds, "pri_outs": pri_outs,
                    "pri_preds": pri_preds}

    def inference(self, model, dataset, return_preds=False):
        eval_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size_per_device * self.device_count,
                                 num_workers=1, shuffle=False)
        model.eval()
        cate_outs, pri_outs = [], []
        cate_gts, cate_preds, pri_gts, pri_preds = [], [], [], []
        with torch.no_grad():
            wrap_loader = tqdm(eval_loader, desc="predicting")
            for j, batch in enumerate(wrap_loader):
                if "categories" in batch:
                    batch.pop("categories")
                    categories_indices = batch.pop("categories_indices").tolist()
                    cate_gts.extend(categories_indices)

                if "priority" in batch:
                    priority = batch.pop("priority")
                    batch.pop("priority_score")
                    pri_gts.extend(priority)
                if "events" in batch:
                    batch.pop("events")
                batch.pop("raw_text")
                inputs = {k: batch[k].to(self.device) for k in batch}
                classification_logits, regression_logits = model(inputs)
                classification_outs = classification_logits.sigmoid()
                cate_outs.extend(classification_outs.tolist())
                cate_pred = (classification_outs > 0.5).int().tolist()
                cate_preds.extend(cate_pred)

                reg_outs = regression_logits.view(-1).sigmoid().tolist()
                pri_outs.extend(reg_outs)
                pri_preds.extend([map_pri(score) for score in reg_outs])

            return_dict = {}
            if len(cate_gts) == len(cate_preds):
                self.logger.info(f"performance reporting on information types categorisation: ")
                self.logger.info(classification_report(cate_gts, cate_preds, digits=4, target_names=self.cate_classes))
                self.logger.info(f"accuracy score: {accuracy_score(cate_gts, cate_preds)}")
                return_dict.update({"InfoType": calculate_perf(cate_preds, cate_gts)})
                self.logger.info(json.dumps(return_dict, indent=2))

            if len(pri_gts) == len(pri_preds):
                self.logger.info(f"performance reporting on priority estimation: ")
                self.logger.info(classification_report(pri_gts, pri_preds, digits=4))
                self.logger.info(f"accuracy score: {accuracy_score(pri_gts, pri_preds)}")
                return_dict.update({"Priority": calculate_perf(pri_preds, pri_gts)})
                self.logger.info(json.dumps(return_dict, indent=2))
            if return_preds:
                return cate_outs, cate_preds, pri_outs, pri_preds
            model_short_name = self.args.base_model_path_or_name.split('/')[-1]
            model_short_name = model_short_name if not self.args.eda_aug else model_short_name + "-eda"
            return {model_short_name: return_dict}

    def report_data_stats(self, examples):
        tmp_list = [len(example["text"].split(" ")) for example in examples]
        max_ex_len = max(tmp_list)
        avg_ex_len = np.average(tmp_list)
        self.logger.info('Example max length: {} (words)'.format(max_ex_len))
        self.logger.info('Example average length: {} (words)'.format(avg_ex_len))
        self.logger.info('Example std length: {} (words)'.format(np.std(tmp_list)))
        exceed_count = len([i for i in tmp_list if i > self.args.max_seq_length])
        self.logger.info(
            f'Examples with words beyond max_seq_length ({self.args.max_seq_length}): {exceed_count}/{len(examples)} (examples)')
        self.logger.info("##################################")

    def submit(self, outs, edition="2020bt1", runtag="default"):
        edition = "2020bt1" if edition == "2020b" else edition
        assert edition in task_editions_available, f"please select an edition from: {task_editions_available}"
        examples = outs.pop("examples")
        it_preds = outs.pop("cate_preds")
        it_outs = outs.pop("cate_outs")  # also use it for 2021A
        # pri_preds = outs.pop("pri_preds")
        pri_outs = outs.pop("pri_outs")  #
        event_ids, post_ids = [], []

        events_map = {}
        if edition != "2021a":
            events_map = edition2eventmapping[edition]

        for idx, example in enumerate(examples):
            if edition != "2021a":
                event_ids.append(events_map[example["event_id"].lower()])
            else:
                event_ids.append(example["event_id"])
            post_ids.append(example["post_id"])

        it_pred_labels = []
        for index, it_one_hot in enumerate(it_preds):
            it_pred_labels.append([self.cate_classes[i] for i, each in enumerate(it_one_hot) if each == 1])

        pri_pred_labels = []
        p_finals = []

        priority_from_its = False  # use regression priority works better
        normalized_it2priorityscore = get_normalized_it2priorityscore()
        for index, pri_score in enumerate(pri_outs):
            p_score = pri_score
            if priority_from_its:
                its = it_pred_labels[index]
                if its == ["Irrelevant"] or its == []:
                    p_final = 0.0
                else:
                    # p_final=np.mean([info_type_priority_weight_dict[it.split("-")[-1]] for it in its])
                    # p_final = (np.mean([info_type_priority_weight_dict[it.split("-")[-1]] for it in its]) + p_score) / 2
                    # p_final = (np.mean([normalized_it2priorityscore[it] for it in its]) + p_score) / 2
                    p_final = max([normalized_it2priorityscore[it] for it in its])
            else:
                p_final = p_score
            p_finals.append(p_final)
            if p_final > 0.75:
                pri_pred_labels.append("Critical")
            elif p_final > 0.5 and p_final <= 0.75:
                pri_pred_labels.append("High")
            elif p_final > 0.25 and p_final <= 0.5:
                pri_pred_labels.append("Medium")
            else:
                pri_pred_labels.append("Low")

        assert len(event_ids) == len(post_ids) == len(it_pred_labels) == len(pri_pred_labels) == len(p_finals)
        sub_dict = {}
        for index, it_labels in enumerate(it_pred_labels):
            post_id = post_ids[index]
            p_label = pri_pred_labels[index]
            event_label = event_ids[index]
            p_final = p_finals[index]

            if it_labels == []:
                it_labels = ["Irrelevant"]

            if edition != "2021a":
                numeric_it_pred = [0.0] * len(self.cate_classes)
                for each in it_labels:
                    numeric_it_pred[its_list.index(each)] = 1.0
            else:
                numeric_it_pred = [0] * len(self.cate_classes)
                for each in it_labels:
                    numeric_it_pred[its_list.index(each)] = 1

            if edition != "2021a":
                line_sub = event_label + "\tQ0\t" + str(post_id) + "\t" + "#" + "\t" + str(
                    self.pri_map[p_label]) + "\t" + str(numeric_it_pred) + "\t" + runtag + "\n"
            else:
                it_scores = [round(s, 4) for s in it_outs[index]]
                line_sub = json.dumps(
                    {"topic": event_label, "runtag": runtag, "tweet_id": str(post_id), "priority": round(p_final, 4),
                     "info_type_scores": it_scores, "info_type_labels": numeric_it_pred}) + "\n"
                # line_sub = line_sub.replace(", ",",")
            if event_label not in sub_dict:
                sub_dict[event_label] = {line_sub: p_final}
            else:
                sub_dict[event_label][line_sub] = p_final

        submit_content = ""
        for each in sub_dict:
            rank_inevent = 1
            sub_inevent = sub_dict[each]
            sub_inevent_sorted = sorted(sub_inevent.items(), key=lambda kv: kv[1], reverse=True)

            for tup in sub_inevent_sorted:
                if edition != "2021a":
                    # replace "#" with rank index
                    line_sub = tup[0].replace("#", str(rank_inevent))
                else:
                    line_sub = tup[0]
                submit_content += line_sub
                rank_inevent += 1

        if not os.path.isdir(os.path.join(self.args.data_path, "subs")):
            os.makedirs(os.path.join(self.args.data_path, "subs"), exist_ok=True)

        submit_path = os.path.join(self.args.data_path, "subs", runtag)
        with open(submit_path, "w") as f:
            f.write(submit_content)

        # gzip submission
        import gzip
        f_in = open(submit_path, "rb")
        gzip_submit_path = os.path.join(self.args.data_path, "subs", runtag + ".gz")
        f_out = gzip.open(gzip_submit_path, 'wb')
        f_out.write(f_in.read())
        f_out.close()
        f_in.close()
        self.logger.info(f"Find the generated submission in: {submit_path}")
        self.logger.info(f"Find the generated submission (gzipped) in: {gzip_submit_path}")

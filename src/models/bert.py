import torch.nn as nn
import torch
from transformers import BertModel, BertConfig
from src.utils import config



class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=config.hidden_size,
                      out_channels=config.num_filters,
                      kernel_size=kernel_size) for kernel_size in config.kernel_sizes])
        self.classifier = nn.Linear(config.num_filters*len(config.kernel_sizes), config.num_classes)
        # 初始化
        # self.apply(self.init_bert_weights)
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, embeddings, labels=None):
        # embeddings: [batch_size, seq_len, bert_dim = 768]
        encoded_layers = self.dropout(embeddings)
        encoded_layers = encoded_layers.permute(0, 2, 1)
        conv_outputs = [conv(encoded_layers) for conv in self.convs] # n_filter个[batch_size,num_filter=256, *]
        conv_outputs = [nn.functional.relu(conv_output) for conv_output in conv_outputs]
        pooled = [nn.functional.max_pool1d(conv_output, conv_output.shape[2]).squeeze(2) for conv_output in conv_outputs]
        # poold # filter_num个 [batch_num, num_filter=256]
        cat = self.dropout(torch.cat(pooled, dim=1))  # cat: [batch_size, filter_num * len(filter_sizes)]
        logits = self.classifier(cat)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
            return loss
        else:
            return logits


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(
            config.bert_path, num_labels=config.num_classes)
        dropout = config.dropout  # 丢弃
        self.bert = BertModel.from_pretrained(config.bert_path,
                                              config=model_config)
        # 冻结bert设置
        freeze_bert = config.freeze_bert  # 是否冻结BERT
        freeze_layers = config.freeze_layers
        if freeze_bert != "":
            for name, param in self.bert.named_parameters():
                if freeze_bert == "all" or name.startswith(tuple(freeze_bert)):
                    param.requires_grad = False
                    print(f"Froze layer {name}...")
        # freeze_layers is a string "1,2,3" representing layer number
        if freeze_layers != "":
            layer_indexes = [int(x) for x in freeze_layers.split(",")]
            for layer_idx in layer_indexes:
                for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                    print("Froze Layer: ", layer_idx)

        # fine-tune设置
        # 使用textcnn
        if config.textcnn:
            print(f"setting textcnn")
            self.textcnn = TextCNN(config)
        # 使用线性层
        if config.linear:
            self.fc = nn.Linear(768,14)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        if config.linear:
            _, pooled = self.bert(context,
                                  attention_mask=mask,
                                  token_type_ids=token_type_ids)
            out = self.fc(pooled)

        elif config.textcnn:
            last_hidden_state, _ = self.bert(context,
                                             attention_mask=mask,
                                             token_type_ids=token_type_ids)
            out = self.textcnn(last_hidden_state)
            
        return out



if __name__ == '__main__':
    cnn = TextCNN(config)
    embedding = torch.randn((2, 100, 768))
    print(embedding.shape)
    print(cnn(embedding))










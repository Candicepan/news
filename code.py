#!/usr/bin/env python
# coding: utf-8

# ## 题目1：查看Paddle、PaddleHub版本及模型信息（6分）
# 
# 模型信息  
# 
# ![image.png](attachment:029d414d-8b90-437e-a402-9c65efa371ba.png)

# In[3]:


import paddle
import paddlehub as hub

print("paddle版本号为： ",paddle.__version__)
print("paddlehub版本号为： ",hub.__version__)


# ## 题目2：打印THUCNews验证集前3条、后3条数据 

# In[4]:


# 进入到这个文件夹下
get_ipython().run_line_magic('cd', '/home/aistudio/data/data16287/')
# 解压数据集
get_ipython().system('tar -zxvf thu_news.tar.gz')
# 查看当前文件夹下的内容
get_ipython().system('ls -hl thu_news')


# In[10]:


print("前三条：\n")
get_ipython().system('head -n 3 thu_news/valid.txt')

print("\n后三条：\n")
get_ipython().system('tail -n 3 thu_news/valid.txt')


# ## 题目3：完整跑通基于PaddleHub的新闻文本分类任务

# In[11]:


# 导入负责文件处理的Python包
import os
import io
import csv

# 导入数据集占位符
from paddlehub.datasets.base_nlp_dataset import InputExample, TextClassificationDataset

# 定义模型
label=['财经', '彩票', '房产', '股票', '家居' , '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']

#选择模型
model = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=len(label))

# 数据集存放位置
DATA_DIR="/home/aistudio/data/data16287/thu_news"

class ThuNews(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'valid.txt'
        super(ThuNews, self).__init__(
            base_path=DATA_DIR,
            data_file=data_file,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            is_file_with_header=True,
            label_list=label)

    # 解析文本文件里的样本
    def _read_file(self, input_file, is_file_with_header: bool = False):
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, text_a=line[0], label=line[1])
                    seq_id += 1
                    examples.append(example)
                return examples

train_dataset = ThuNews(model.get_tokenizer(), mode='train', max_seq_len=128)
dev_dataset = ThuNews(model.get_tokenizer(), mode='dev', max_seq_len=128)
test_dataset = ThuNews(model.get_tokenizer(), mode='test', max_seq_len=128)

# 打印后三行
for e in train_dataset.examples[:3]:
    print(e)


# In[12]:


import paddle

# 定义一个优化器，相当于老师
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())  # 优化器的选择和参数配置
# 定义个训练器，相当于考试+课后复习的流程
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt', use_gpu=True)        # fine-tune任务的执行者
# 开始训练 开始考试+复习
trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, save_interval=1)   # 配置训练参数，启动训练，并指定验证集


# In[13]:


data = [
    # 房产
    ["昌平京基鹭府10月29日推别墅1200万套起享97折  新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。  京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。"],
    # 游戏
    ["尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。  这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。  片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。"],
    # 体育
    ["罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪  记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。  拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。"],
    # 教育
    ["新总督致力提高加拿大公立教育质量  滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。  约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。"]
]

label_list=['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']
label_map = { 
    idx: label_text for idx, label_text in enumerate(label_list)
}

model = hub.Module(
    name='ernie',
    task='seq-cls',
    load_checkpoint='./ckpt/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))


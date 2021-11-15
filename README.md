# news

## 题目1：查看Paddle、PaddleHub版本及模型信息（6分）

模型信息  

![image](https://user-images.githubusercontent.com/48274303/141740090-f777b59c-b9e4-4100-ac21-f8d7147ce3fc.png)


```python
import paddle
import paddlehub as hub

print("paddle版本号为： ",paddle.__version__)
print("paddlehub版本号为： ",hub.__version__)
```

    paddle版本号为：  2.2.0
    paddlehub版本号为：  2.0.4


## 题目2：打印THUCNews验证集前3条、后3条数据 


```python
# 进入到这个文件夹下
%cd /home/aistudio/data/data16287/
# 解压数据集
!tar -zxvf thu_news.tar.gz
# 查看当前文件夹下的内容
!ls -hl thu_news
```

    /home/aistudio/data/data16287
    thu_news/
    thu_news/test.txt
    thu_news/valid.txt
    thu_news/train.txt
    总用量 30M
    -rw-r--r-- 1 aistudio aistudio 3.7M 11月 19  2019 test.txt
    -rw-r--r-- 1 aistudio aistudio  23M 11月 19  2019 train.txt
    -rw-r--r-- 1 aistudio aistudio 3.6M 11月 19  2019 valid.txt



```python
print("前三条：\n")
!head -n 3 thu_news/valid.txt

print("\n后三条：\n")
!tail -n 3 thu_news/valid.txt
```

    前三条：
    
    text_a	label
    直击特里勾手助小牛反超神鸟发威火箭仍处劣势　　新浪体育讯　北京时间4月16日消息，火箭今天迎来常规赛的收官战。客场挑战达拉斯小牛的比赛将关系到火箭队最终的排名，目前西部的竞争仍然非常激励。尤其是西部第二到第七的六支球队将在今天展开捉对厮杀的情况下，黄蜂、小牛、开拓者以及马刺都有可能是火箭队的下一个对手。以下为本场比赛的精彩瞬间——　　第四节比赛还剩下8分多钟，姚明重新回到场上，但是小牛队的霍里随后在右翼接队友传球后上演一记三分跳投，虽然身体动作已经变形，但是他仍然将球命中，随后基德在弧顶处的一记三分跳投帮助小牛终于将比分反超，火箭在进攻中直接打不开局面，而小牛更是利用特里空切后的一记勾手将比分反超至4分，包括库班在内的所有小牛现场球迷都站起来振臂高呼，而此时阿德尔曼还在场边低头深沉的思索中。　　比赛还剩下最后5分多钟，洛瑞带球突破中被小牛队员犯规，洛瑞来不及刹车让自己直接撞到了观众席上，右腿疼痛难忍的他脸都几乎变形，结果他两罚两中，火箭队还落后两分。随后，洛瑞将球直接传给此时已经回到场上的阿泰，野兽此时将球稳下，并没有急于进攻，随后他顺势将球塞给兰德里，神鸟利用对方站位空隙直接杀到篮下，上篮成功，火箭将比分终于追至80平。场上局面对于火箭来说绝对是不容有失。　　(sabrina)	体育
    北京网购年消费额112亿元　　商报讯(记者吴文治)昨日，淘宝网发布的《2009-2010年度中国网购热门城市报告》显示，北京年度网购消费力达112.5亿元，与上海相差近62亿元，位列十大热门消费力城市第二位。此外，男性网购的消费金额高出女性，与“女性是网购主力军”的传统观念不符。　　淘宝公布的报告显示，中国网购消费力十大城市分别是上海、北京、深圳、杭州、广州、南京、苏州、天津、温州和宁波。主要集中在以江浙沪为主的长三角地区、以广深为主的珠三角地区和以北京为主的京津地区。北京年度网购112.5亿元的消费额，占国内城市网购消费额的5.6%。　　中国网购消费力十大城市的消费金额性别来源比例中，男性占比超过了女性。前者占比达到53.5%，后者则为46.5%。不过，在成交人数、成交笔数等关键数据上显示，女性消费者均高于男性。此外，在十大网购热门城市中，25岁-34岁的群体成为网购消费的主力军，占比达到62.49%。	科技
    
    后三条：
    
    事业测试：你工作易受他人干扰吗(图)　　独家撰稿：苑椿　心理测试征稿启事 欢迎关注新浪星座微博　　办公室永远是个龙蛇混杂、藏龙卧虎的地方，你永远不知道一张张面具底下会是怎样的脸庞，你是否还傻傻的对别人的话听之任之，完全搞乱了自己工作的步调？还是笃定的坚守阵地，从未被谣言动摇分毫？赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    趣味测试：你怎么红红火火过春节(图)　　独家撰稿：岚　心理测试征稿启事 欢迎关注新浪星座微博　　红红火火过大年啦，每年的此时你都会如何度过呢？是跟家人爱人在一起还是跟朋友兄弟外出欢聚呢？亦或背起行囊远离嘈杂，无论如何，总会有一种适合你的方式，赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    人际测试：你的人际磁场强大吗(图)　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。	星座


## 题目3：完整跑通基于PaddleHub的新闻文本分类任务


```python
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
```

    INFO:filelock:Lock 140344254516176 acquired on /home/aistudio/.paddlehub/tmp/ernie_tiny


    Download https://bj.bcebos.com/paddlehub/paddlehub_dev/ernie_tiny_2.0.1.tar.gz
    [##################################################] 100.00%
    Decompress /home/aistudio/.paddlehub/tmp/tmpu70sxlx0/ernie_tiny_2.0.1.tar.gz
    [##################################################] 100.00%


    [2021-11-15 14:50:28,767] [    INFO] - Successfully installed ernie_tiny-2.0.1
    INFO:filelock:Lock 140344254516176 released on /home/aistudio/.paddlehub/tmp/ernie_tiny
    [2021-11-15 14:50:28,772] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/ernie_tiny.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-tiny
    [2021-11-15 14:50:28,775] [    INFO] - Downloading ernie_tiny.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/ernie_tiny.pdparams
    100%|██████████| 354158/354158 [00:11<00:00, 30088.26it/s]
    W1115 14:50:40.634047   217 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1115 14:50:40.639243   217 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [2021-11-15 14:50:48,276] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/vocab.txt and saved to /home/aistudio/.paddlenlp/models/ernie-tiny
    [2021-11-15 14:50:48,284] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/vocab.txt
    100%|██████████| 459/459 [00:00<00:00, 2811.42it/s]
    [2021-11-15 14:50:48,527] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/spm_cased_simp_sampled.model and saved to /home/aistudio/.paddlenlp/models/ernie-tiny
    [2021-11-15 14:50:48,530] [    INFO] - Downloading spm_cased_simp_sampled.model from https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/spm_cased_simp_sampled.model
    100%|██████████| 1083/1083 [00:00<00:00, 4031.66it/s]
    [2021-11-15 14:50:48,868] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/dict.wordseg.pickle and saved to /home/aistudio/.paddlenlp/models/ernie-tiny
    [2021-11-15 14:50:48,871] [    INFO] - Downloading dict.wordseg.pickle from https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/dict.wordseg.pickle
    100%|██████████| 161822/161822 [00:05<00:00, 29210.70it/s]
    [2021-11-15 14:52:09,038] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/vocab.txt
    [2021-11-15 14:52:09,041] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/spm_cased_simp_sampled.model
    [2021-11-15 14:52:09,043] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/dict.wordseg.pickle
    [2021-11-15 14:52:23,811] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/vocab.txt
    [2021-11-15 14:52:23,814] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/spm_cased_simp_sampled.model
    [2021-11-15 14:52:23,816] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-tiny/dict.wordseg.pickle


    text=主旋律在突破中赢得掌声江青正面形象亮相荧屏　　新中国成立60周年之际，银幕、荧屏、舞台上，各种献礼作品精彩纷呈。这些以历史为依据的艺术作品，或者人性化地展现伟人的性格，或者客观地诠释有争议性的人物，或者不回避历史，较以往艺术创作都有很大突破，而观众也看得非常过瘾。　　文/记者周娴、莫斯其格、张素芹　　很客观《解放》还原“争议”人物　　电视剧《解放》追求全景式展现历史，真实还原人物的宗旨，对很多颇受争议的历史人物做了客观的展现，江青首次以青春美好的正面形象出现在荧屏上。同时，该剧用大量篇幅表现了国民党中一些能征善战的将领，客观描绘了他们的性格特征，展现了他们作为军人的英勇与悲壮。相对于此类作品以往的“脸谱化”表现而言，这是一个突破。　　首次看到江青的正面形象　　在《解放》中，江青30多年来首次以正面形象出现在荧屏上，刚刚踏入解放战争洪流的江青，活泼开朗的性格被重点放大。对丈夫的崇拜和深情，照顾病中周恩来的细心周到，这些情节，都将少女江青的情怀刻画得淋漓尽致。剧情进入到中段，她与毛泽东及女儿的温馨生活场面则成为惨烈战争的调和剂。　　网友意见：《解放》里江青有大量的戏，她看上很纯很天真，这可是第一次啊。只有这样才能显得作品真实可信，应该客观公正地评价每一个人。　　主创回应：导演唐国强说，《解放》是站在唯物主义的历史观上，在特定的革命阶段，展现人物固有性格特征，还原人物的本来面貌。　　林彪与将士高唱军歌　　林彪是一代军事家，十大元帅中最年轻的元帅。10年前《大决战》中，首度出现林彪的正面形象，运筹帷幄，决胜千里。《解放》中，又让观众看到了林彪与将士们一起高唱“向前向前向前，我们的队伍向太阳”的景象，看到了一个更加不一样的林彪。　　网友意见：解放战争林彪功不可没，电视剧应该好好表现他。“千秋功罪，自有后人评说”。《解放》对林彪有了比较客观的评价，演员的诠释也相当出色，完全表现出一个具有卓越军事才能、个性孤僻又服从大局的军事家的感觉。　　主创回应：林彪的饰演者由立平说，为了接近林彪的身材，本来体重就不足130斤的我还是咬牙减去了10斤。保持在这个体重我心里才踏实，不光形体像了，而且通过减肥，似乎更能感受到他当时的内心压力。我对军事家有着独特的情感，我的这份情感不单单是向一个人，而是向整个老一辈军事家——致敬！　　国民党将领的铁骨柔情　　剧情进入中段后，《解放》将大量笔墨放在了描写战争中的传奇将领上。张灵甫形象英俊，气质儒雅，是国民党陆军中将、抗日名将，蒋介石对其厚爱有加。他死于国共内战时期的孟良崮战役，孟良崮战役中，对张灵甫采用了正面而又全面的描述，突出了他高傲忠诚的性格，也表现了下属对他的爱戴。　　“干城之将”陈明仁是另一个被正面渲染的国民党将领。“四平之战”中，陈明仁的坚毅、英勇与调兵遣将的能力得到了最好的展现，性格刚烈，悲情又悲壮。　　网友意见：《解放》胜在细节，陈明仁剃发明志和拼死一搏之前留下遗书的段落，已超越了国共战争的敌我界限，成为一个中国军人铁骨柔情的最佳诠释。　　主创回应：唐国强说：“越是客观，便越能感受到伟大。《解放》更注重还原历史，它甚至比《建国大业》更写实，我相信这是最能打动观众的。”　　人性化《建国大业》鲜活伟人　　在以往讲述领袖的影视作品中，领袖形象都会被塑造得笑容可掬、和蔼可亲，但在影片《建国大业》中毛泽东烟不离手，周恩来发起了脾气，几个领袖在一起醉酒唱歌，这些细节完全摆脱了以往主旋律作品中领袖人物概念化的形象，从细节还原了一个个有七情六欲人性化的领袖形象。　　场景一：怒发冲冠　　得知冯玉祥在海上遇难后，毛泽东气得踢翻了水盆，周恩来更大骂手下“都是猪脑袋”，“你们谁能负这个责”，不仅摔了文件夹，还摔了茶杯，连衣服都迸开了。　　网友意见：这是《建国大业》中对周恩来的刻画最有力的一笔，“以前对总理的印象总停留在‘温文尔雅’四个字上，原来总理也跟普通人一样是有脾气的。”“主席突然踢翻了洗脚盆，哐的一声，吓我一跳。但想想，又很符合逻辑。”　　主创说法：周恩来的扮演者刘劲说：“领袖也有喜怒哀乐，导演告诉我，放开演，要表现得怒发冲冠，这就突破了领袖人物顾全大局、和蔼可亲的固有形象，使人物显得更加真实可信。　　场景二：醉酒当歌　　接到淮海战役胜利捷报的当晚，中央四位领导人喝酒庆祝，周恩来揽着朱德、刘少奇高唱《国际歌》，喝醉了的毛泽东歪倒在一旁咧嘴直笑，这样“奔放”的中共领袖在以前的主旋律影片中绝无仅有。　　网友意见：影片的这一段拍得相当感人，不少网友留言称被这个场景感动落泪，“我觉得这是影片最叫人难忘的一幕，我一大老爷们也跟着唱起来了，同时眼眶也湿润了……”“从这一段中能体会领导人的真性情，给人一种亲近感。”　　主创回应：在以往的主旋律电影里没有出现过这种有“丑化”领导人嫌疑的镜头。黄建新解释说，这一段的确是艺术发挥，经过剧组讨论，觉得这样更能显示出领导人在“天下初定”时的那种畅快和浪漫的革命情感，而这一段落在审查时，同样没有收到任何修改建议。　　场景三：“有气无力”喊口号　　《建国大业》中最令人觉得意外的地方，莫过于结尾时毛主席在政协会议上发言，结束时所喊的口号“中华人民共和国万岁”、“联合政府万岁”等。按照以往主旋律电影的处理方式，这种口号无疑要喊得一个比一个响亮，但唐国强扮演的毛泽东，在这里的处理却是一个比一个声音弱，显得有些“有气无力”。　　网友意见：以往的主旋律影片中，主席都是慷慨激昂地喊口号，振奋人心，想不通《建国大业》为什么要这样处理。　　主创回应：导演黄建新解释说，这是唐国强自己提出来的，因为他们在研究了历史上毛主席的演讲录音，发现当年毛主席就是这样的口气，“其实最重要的不是语气，而是毛主席演讲完毕之后，慢慢抬头环顾现场。以往他演讲都是对党内人士，这一次是真正面对全中国在政治方面最有影响力的人。他最后的表情，可谓意味深长。”　　场景四：主席遭轰炸　　毛主席在河北城南庄被潜伏特务告密，蒋介石派两架飞机轰炸城南庄。毛主席当时吃了安眠药刚睡下，警卫员一起抬着担架转移身穿睡衣的主席，最终化险为夷。　　网友意见：这一段惊险、紧张，意外的是主席“临危不乱”的时候居然身穿睡衣！　　主创回应：导演黄建新解释：“这次突破地方就是想让镜头深入到这些历史人物的内心，希望带来新鲜感。”　　不回避　　《复兴之路》用4分钟讲述“文革”　　大型音乐舞蹈史诗《复兴之路》，是继《东方红》、《中国革命之歌》之后，中国文化艺术史上的又一部大型音乐舞蹈史诗。在两个半小时的演出中，八一电影制片厂副厂长刘星创作的朗诵诗《沉思与抉择》用诗歌朗诵的方式讲述了“文革”的十年历史，时间长约4分钟，这也是第一次在中国庆典舞台上涉及“文革”话题。　　场景：首次涉及“文革”　　“如果不是为寻找历史是在哪里转弯，如果不是为寻找民族复兴新的起点，谁还愿意揭开往日的伤口，谁还想回首那曾经的劫难……”在这首诗中，没有出现“文革”的字眼，但是，隐喻、象征的手法，使得交代明确、措辞得体。结尾的处理更是“神来之笔”———原本雷声滚滚、乌云遮空的天空明朗起来，“人们听到中国共产党人的浩然之声，又一次在大地和天空间回旋。”　　网友意见：　　《复兴之路》“结合历史，紧跟时代，用音乐、舞蹈、朗诵、合唱等表演形式，用两个半小时的时间，浓缩了169年的中国历史。这是一次心灵的洗礼，激起了人们的情感共鸣。是最为完美的精彩演出。”　　主创回应：　　对于“文革”等历史细节的出现，文化部部长蔡武接受媒体采访时表示，《复兴之路》涉及这段历史是“因为如果没有对‘文化大革命’这一段的一个交代、一个表现，那么人们就会问，那么三中全会的意义在什么地方，为什么它是一个历史的转折，为什么它提出了改革开放的政策，它的针对性是什么，就是要讲清楚这个问题所以必须要讲到‘文化大革命’，所以我们不回避这个问题，但是我们又不去渲染它”。　　诗作者刘星严格遵循“不回避，不渲染”的创作原则，但用何种方式展开叙述，让他冥思苦想了几个月，直到想到了古典诗词中“比兴”的艺术手法。他接受采访的时候表示：“这二十几行诗，我改了二十多遍。阎肃老师也帮助我一起改写，经常为一两个字绞尽脑汁。我们共同把这个任务完成好”。　　《复兴之路》的总导演张继刚在接受采访的时候表示，对历史上的事情的态度应该是不回避也不渲染，而也正是因为这段历史、一系列的曲折，“让我们更珍惜此时此刻——中国历史上最好的时期”。　　诗歌朗诵的表演者瞿弦和接受采访的时候称，他们这代人都经历了“文革”，在“文革”中有着不同的遭遇，在每个人一生中都留下了不可磨灭的记忆，所以作为表演者要“用自己的体会来感悟，来体现我们多么珍惜这个转折”。	label=娱乐
    text=骑士总经理位置恐难保布朗下课后首发声明谢球队　　新浪体育讯　北京时间5月27日消息，自从被骑士解雇后，迈克-布朗还未在公开场合发表过任何言论。但当地时间周三，布朗委托骑士对外发表了一份声明，并在声明中感谢克利夫兰给了他这次执教NBA球队的经历。在文章的最后，布朗甚至还动情的表示，过去在骑士队中的5年，对于他来说有着“特殊的经历”。　　布朗在声明中写道：“我为在克利夫兰拥有5年的执教经历感到骄傲。很荣幸能与这么多尊重我的球员和员工共事，我很享受和他们在一起奋斗的时光。同时，我还要感谢球队老板丹-吉尔伯特对我的奉献和鼎力支持。虽然直到我告别这支球队时，我仍然没有为这座城市带来他们想要的东西，但我仍然对在这里的执教成绩感到满意和自豪。”　　在过去的两个赛季中，布朗率领的骑士共赢得了127场常规赛的胜利。而布朗本人也曾经在2009年凭借优异的率队成绩，荣膺NBA联盟最佳教练员称号。而布朗带领骑士所取得的最好成绩，当属2007年闯入最终的总决赛。　　在被骑士解雇之后，布朗一直赋闲在家。不过也许在不久之后，布朗就将迎来下一份主教练的工作。因为在目前联盟的30支球队中，仍然有5支球队在为新赛季主教练的人选发愁，而布朗也极有可能成为这5支球队选帅的潜在目标。　　而据《克利夫兰平原经销商》报道，解雇布朗之后，并不意味着骑士内部的人员调整就此告一段落，反之随着布朗的下课，球队总经理丹尼-费里的前景也变得扑朔迷离。而一旦费里离开，那球队总经理的职位，就将由骑士助理教练克里斯-格兰特接替。　　费里的合同6月30日就将到期，不过从解雇布朗的举动来看，费里也极有可能选择继续留守克利夫兰。而一旦费里最终留任，那毫无疑问，骑士最重要的事情就是尽可能地留下“小皇帝”勒布朗-詹姆斯。　　(小林)	label=体育
    text=国足11月西亚行会中超老友鲁能新援鸿门宴意欲何为　　记者张远济南报道由于在亚洲杯预赛上，中国和黎巴嫩同在D组，11月14日和22日，中国队将先客后主迎战黎巴嫩。小组赛前两场黎巴嫩一分未得，出线希望渺茫，对此安塔尔并不太在意，他更关注鲁能的队友谁有可能进入国家队，“在我看来，至少有4名队员可以进入国家队，如果他们到时候能到叙利亚比赛，对我来说那别提是一件多么开心的事情了，到时候我会邀请他们品尝叙利亚的美味，也许会请他们到我家做客，我很乐意在厨房里露上一手。”　　安塔尔的哥哥菲塞尔也是一名职业球员，“他在黎巴嫩国家队打左边后卫，到比赛时他也会来，我和哥哥一起代表黎巴嫩，鲁能的队友代表中国，这一定很有趣。”安塔尔这样说并不意味着他放弃了亚洲杯出线的希望，“我们前两场比赛都输了，形势不妙，但我们不会轻易放弃，对中国队的比赛一定会全力以赴。”　　安塔尔说，他最希望黎巴嫩和中国队携手出线，但这实现的可能性几乎为零。“但不管怎么说，过去的8年时间里，我为我的国家付出了努力，我打进了28个进球，我认为自己的成就非常值得骄傲，将来在鲁能，我也期待能够用自己的努力为球队带来荣誉。”	label=体育



```python
import paddle

# 定义一个优化器，相当于老师
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())  # 优化器的选择和参数配置
# 定义个训练器，相当于考试+课后复习的流程
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt', use_gpu=True)        # fine-tune任务的执行者
# 开始训练 开始考试+复习
trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, save_interval=1)   # 配置训练参数，启动训练，并指定验证集
```

    [2021-11-15 15:09:45,941] [ WARNING] - PaddleHub model checkpoint not found, start from scratch...
    [2021-11-15 15:09:47,045] [   TRAIN] - Epoch=1/3, Step=10/282 loss=2.2518 acc=0.3125 lr=0.000050 step/sec=9.09 | ETA 00:01:33
    [2021-11-15 15:09:47,971] [   TRAIN] - Epoch=1/3, Step=20/282 loss=1.6162 acc=0.5781 lr=0.000050 step/sec=10.80 | ETA 00:01:25
    [2021-11-15 15:09:48,898] [   TRAIN] - Epoch=1/3, Step=30/282 loss=1.1932 acc=0.7031 lr=0.000050 step/sec=10.79 | ETA 00:01:23
    [2021-11-15 15:09:49,822] [   TRAIN] - Epoch=1/3, Step=40/282 loss=0.8371 acc=0.7812 lr=0.000050 step/sec=10.82 | ETA 00:01:22
    [2021-11-15 15:09:50,749] [   TRAIN] - Epoch=1/3, Step=50/282 loss=0.6336 acc=0.8406 lr=0.000050 step/sec=10.79 | ETA 00:01:21
    [2021-11-15 15:09:51,671] [   TRAIN] - Epoch=1/3, Step=60/282 loss=0.6424 acc=0.8031 lr=0.000050 step/sec=10.84 | ETA 00:01:20
    [2021-11-15 15:09:52,589] [   TRAIN] - Epoch=1/3, Step=70/282 loss=0.5147 acc=0.8531 lr=0.000050 step/sec=10.89 | ETA 00:01:20
    [2021-11-15 15:09:53,508] [   TRAIN] - Epoch=1/3, Step=80/282 loss=0.5045 acc=0.8469 lr=0.000050 step/sec=10.88 | ETA 00:01:19
    [2021-11-15 15:09:54,431] [   TRAIN] - Epoch=1/3, Step=90/282 loss=0.2912 acc=0.9219 lr=0.000050 step/sec=10.84 | ETA 00:01:19
    [2021-11-15 15:09:55,357] [   TRAIN] - Epoch=1/3, Step=100/282 loss=0.3779 acc=0.8906 lr=0.000050 step/sec=10.79 | ETA 00:01:19
    [2021-11-15 15:09:56,297] [   TRAIN] - Epoch=1/3, Step=110/282 loss=0.3083 acc=0.9125 lr=0.000050 step/sec=10.64 | ETA 00:01:19
    [2021-11-15 15:09:57,231] [   TRAIN] - Epoch=1/3, Step=120/282 loss=0.3696 acc=0.8906 lr=0.000050 step/sec=10.70 | ETA 00:01:19
    [2021-11-15 15:09:58,164] [   TRAIN] - Epoch=1/3, Step=130/282 loss=0.3092 acc=0.9125 lr=0.000050 step/sec=10.72 | ETA 00:01:19
    [2021-11-15 15:09:59,089] [   TRAIN] - Epoch=1/3, Step=140/282 loss=0.2462 acc=0.9313 lr=0.000050 step/sec=10.81 | ETA 00:01:19
    [2021-11-15 15:10:00,012] [   TRAIN] - Epoch=1/3, Step=150/282 loss=0.3250 acc=0.8938 lr=0.000050 step/sec=10.83 | ETA 00:01:19
    [2021-11-15 15:10:00,932] [   TRAIN] - Epoch=1/3, Step=160/282 loss=0.3087 acc=0.9031 lr=0.000050 step/sec=10.86 | ETA 00:01:19
    [2021-11-15 15:10:01,849] [   TRAIN] - Epoch=1/3, Step=170/282 loss=0.3297 acc=0.9000 lr=0.000050 step/sec=10.91 | ETA 00:01:19
    [2021-11-15 15:10:02,768] [   TRAIN] - Epoch=1/3, Step=180/282 loss=0.2511 acc=0.9375 lr=0.000050 step/sec=10.88 | ETA 00:01:19
    [2021-11-15 15:10:03,693] [   TRAIN] - Epoch=1/3, Step=190/282 loss=0.3335 acc=0.9125 lr=0.000050 step/sec=10.81 | ETA 00:01:19
    [2021-11-15 15:10:04,619] [   TRAIN] - Epoch=1/3, Step=200/282 loss=0.2427 acc=0.9344 lr=0.000050 step/sec=10.79 | ETA 00:01:18
    [2021-11-15 15:10:05,549] [   TRAIN] - Epoch=1/3, Step=210/282 loss=0.3038 acc=0.9281 lr=0.000050 step/sec=10.76 | ETA 00:01:18
    [2021-11-15 15:10:06,475] [   TRAIN] - Epoch=1/3, Step=220/282 loss=0.2409 acc=0.9156 lr=0.000050 step/sec=10.79 | ETA 00:01:18
    [2021-11-15 15:10:07,401] [   TRAIN] - Epoch=1/3, Step=230/282 loss=0.2657 acc=0.9250 lr=0.000050 step/sec=10.80 | ETA 00:01:18
    [2021-11-15 15:10:08,326] [   TRAIN] - Epoch=1/3, Step=240/282 loss=0.3089 acc=0.9000 lr=0.000050 step/sec=10.81 | ETA 00:01:18
    [2021-11-15 15:10:09,250] [   TRAIN] - Epoch=1/3, Step=250/282 loss=0.3889 acc=0.8812 lr=0.000050 step/sec=10.82 | ETA 00:01:18
    [2021-11-15 15:10:10,176] [   TRAIN] - Epoch=1/3, Step=260/282 loss=0.2426 acc=0.9219 lr=0.000050 step/sec=10.80 | ETA 00:01:18
    [2021-11-15 15:10:11,103] [   TRAIN] - Epoch=1/3, Step=270/282 loss=0.2695 acc=0.8906 lr=0.000050 step/sec=10.79 | ETA 00:01:18
    [2021-11-15 15:10:12,019] [   TRAIN] - Epoch=1/3, Step=280/282 loss=0.2881 acc=0.9187 lr=0.000050 step/sec=10.92 | ETA 00:01:18
    [2021-11-15 15:10:13,648] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - [Evaluation result] avg_acc=0.8550
    [2021-11-15 15:10:16,759] [    EVAL] - Saving best model to ./ckpt/best_model [best acc=0.8550]
    [2021-11-15 15:10:16,762] [    INFO] - Saving model checkpoint to ./ckpt/epoch_1
    [2021-11-15 15:10:20,835] [   TRAIN] - Epoch=2/3, Step=10/282 loss=0.2086 acc=0.9375 lr=0.000050 step/sec=1.36 | ETA 00:01:41
    [2021-11-15 15:10:21,744] [   TRAIN] - Epoch=2/3, Step=20/282 loss=0.1398 acc=0.9625 lr=0.000050 step/sec=11.00 | ETA 00:01:40
    [2021-11-15 15:10:22,666] [   TRAIN] - Epoch=2/3, Step=30/282 loss=0.2109 acc=0.9500 lr=0.000050 step/sec=10.84 | ETA 00:01:39
    [2021-11-15 15:10:23,594] [   TRAIN] - Epoch=2/3, Step=40/282 loss=0.1605 acc=0.9688 lr=0.000050 step/sec=10.77 | ETA 00:01:38
    [2021-11-15 15:10:24,520] [   TRAIN] - Epoch=2/3, Step=50/282 loss=0.1249 acc=0.9719 lr=0.000050 step/sec=10.81 | ETA 00:01:38
    [2021-11-15 15:10:25,446] [   TRAIN] - Epoch=2/3, Step=60/282 loss=0.1667 acc=0.9437 lr=0.000050 step/sec=10.79 | ETA 00:01:37
    [2021-11-15 15:10:26,377] [   TRAIN] - Epoch=2/3, Step=70/282 loss=0.1589 acc=0.9531 lr=0.000050 step/sec=10.75 | ETA 00:01:37
    [2021-11-15 15:10:27,303] [   TRAIN] - Epoch=2/3, Step=80/282 loss=0.1654 acc=0.9531 lr=0.000050 step/sec=10.80 | ETA 00:01:36
    [2021-11-15 15:10:28,226] [   TRAIN] - Epoch=2/3, Step=90/282 loss=0.1737 acc=0.9469 lr=0.000050 step/sec=10.83 | ETA 00:01:36
    [2021-11-15 15:10:29,154] [   TRAIN] - Epoch=2/3, Step=100/282 loss=0.1294 acc=0.9500 lr=0.000050 step/sec=10.78 | ETA 00:01:35
    [2021-11-15 15:10:30,076] [   TRAIN] - Epoch=2/3, Step=110/282 loss=0.1459 acc=0.9406 lr=0.000050 step/sec=10.84 | ETA 00:01:35
    [2021-11-15 15:10:30,999] [   TRAIN] - Epoch=2/3, Step=120/282 loss=0.1629 acc=0.9594 lr=0.000050 step/sec=10.83 | ETA 00:01:34
    [2021-11-15 15:10:31,918] [   TRAIN] - Epoch=2/3, Step=130/282 loss=0.1322 acc=0.9688 lr=0.000050 step/sec=10.88 | ETA 00:01:34
    [2021-11-15 15:10:32,842] [   TRAIN] - Epoch=2/3, Step=140/282 loss=0.1530 acc=0.9531 lr=0.000050 step/sec=10.82 | ETA 00:01:34
    [2021-11-15 15:10:33,764] [   TRAIN] - Epoch=2/3, Step=150/282 loss=0.1762 acc=0.9437 lr=0.000050 step/sec=10.85 | ETA 00:01:33
    [2021-11-15 15:10:34,691] [   TRAIN] - Epoch=2/3, Step=160/282 loss=0.1092 acc=0.9625 lr=0.000050 step/sec=10.78 | ETA 00:01:33
    [2021-11-15 15:10:35,623] [   TRAIN] - Epoch=2/3, Step=170/282 loss=0.1199 acc=0.9625 lr=0.000050 step/sec=10.73 | ETA 00:01:32
    [2021-11-15 15:10:36,556] [   TRAIN] - Epoch=2/3, Step=180/282 loss=0.1695 acc=0.9625 lr=0.000050 step/sec=10.72 | ETA 00:01:32
    [2021-11-15 15:10:37,488] [   TRAIN] - Epoch=2/3, Step=190/282 loss=0.1655 acc=0.9531 lr=0.000050 step/sec=10.73 | ETA 00:01:32
    [2021-11-15 15:10:38,414] [   TRAIN] - Epoch=2/3, Step=200/282 loss=0.1509 acc=0.9563 lr=0.000050 step/sec=10.80 | ETA 00:01:32
    [2021-11-15 15:10:39,337] [   TRAIN] - Epoch=2/3, Step=210/282 loss=0.1422 acc=0.9500 lr=0.000050 step/sec=10.83 | ETA 00:01:31
    [2021-11-15 15:10:40,269] [   TRAIN] - Epoch=2/3, Step=220/282 loss=0.1791 acc=0.9437 lr=0.000050 step/sec=10.74 | ETA 00:01:31
    [2021-11-15 15:10:41,204] [   TRAIN] - Epoch=2/3, Step=230/282 loss=0.1246 acc=0.9688 lr=0.000050 step/sec=10.69 | ETA 00:01:31
    [2021-11-15 15:10:42,135] [   TRAIN] - Epoch=2/3, Step=240/282 loss=0.1062 acc=0.9688 lr=0.000050 step/sec=10.74 | ETA 00:01:31
    [2021-11-15 15:10:43,065] [   TRAIN] - Epoch=2/3, Step=250/282 loss=0.1325 acc=0.9750 lr=0.000050 step/sec=10.76 | ETA 00:01:30
    [2021-11-15 15:10:43,994] [   TRAIN] - Epoch=2/3, Step=260/282 loss=0.1268 acc=0.9625 lr=0.000050 step/sec=10.76 | ETA 00:01:30
    [2021-11-15 15:10:44,924] [   TRAIN] - Epoch=2/3, Step=270/282 loss=0.1263 acc=0.9656 lr=0.000050 step/sec=10.75 | ETA 00:01:30
    [2021-11-15 15:10:45,849] [   TRAIN] - Epoch=2/3, Step=280/282 loss=0.1293 acc=0.9625 lr=0.000050 step/sec=10.82 | ETA 00:01:30
    [2021-11-15 15:10:47,483] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - [Evaluation result] avg_acc=0.9064
    [2021-11-15 15:10:51,251] [    EVAL] - Saving best model to ./ckpt/best_model [best acc=0.9064]
    [2021-11-15 15:10:51,254] [    INFO] - Saving model checkpoint to ./ckpt/epoch_2
    [2021-11-15 15:10:55,266] [   TRAIN] - Epoch=3/3, Step=10/282 loss=0.1118 acc=0.9688 lr=0.000050 step/sec=1.27 | ETA 00:01:42
    [2021-11-15 15:10:56,190] [   TRAIN] - Epoch=3/3, Step=20/282 loss=0.0659 acc=0.9812 lr=0.000050 step/sec=10.82 | ETA 00:01:41
    [2021-11-15 15:10:57,128] [   TRAIN] - Epoch=3/3, Step=30/282 loss=0.0397 acc=0.9969 lr=0.000050 step/sec=10.66 | ETA 00:01:41
    [2021-11-15 15:10:58,057] [   TRAIN] - Epoch=3/3, Step=40/282 loss=0.0588 acc=0.9812 lr=0.000050 step/sec=10.77 | ETA 00:01:41
    [2021-11-15 15:10:58,982] [   TRAIN] - Epoch=3/3, Step=50/282 loss=0.0898 acc=0.9750 lr=0.000050 step/sec=10.81 | ETA 00:01:40
    [2021-11-15 15:10:59,910] [   TRAIN] - Epoch=3/3, Step=60/282 loss=0.0599 acc=0.9812 lr=0.000050 step/sec=10.78 | ETA 00:01:40
    [2021-11-15 15:11:00,834] [   TRAIN] - Epoch=3/3, Step=70/282 loss=0.0664 acc=0.9844 lr=0.000050 step/sec=10.82 | ETA 00:01:39
    [2021-11-15 15:11:01,752] [   TRAIN] - Epoch=3/3, Step=80/282 loss=0.0632 acc=0.9812 lr=0.000050 step/sec=10.89 | ETA 00:01:39
    [2021-11-15 15:11:02,672] [   TRAIN] - Epoch=3/3, Step=90/282 loss=0.0591 acc=0.9844 lr=0.000050 step/sec=10.87 | ETA 00:01:39
    [2021-11-15 15:11:03,596] [   TRAIN] - Epoch=3/3, Step=100/282 loss=0.0541 acc=0.9875 lr=0.000050 step/sec=10.82 | ETA 00:01:38
    [2021-11-15 15:11:04,522] [   TRAIN] - Epoch=3/3, Step=110/282 loss=0.0999 acc=0.9781 lr=0.000050 step/sec=10.79 | ETA 00:01:38
    [2021-11-15 15:11:05,450] [   TRAIN] - Epoch=3/3, Step=120/282 loss=0.0548 acc=0.9812 lr=0.000050 step/sec=10.77 | ETA 00:01:38
    [2021-11-15 15:11:06,381] [   TRAIN] - Epoch=3/3, Step=130/282 loss=0.0538 acc=0.9906 lr=0.000050 step/sec=10.75 | ETA 00:01:38
    [2021-11-15 15:11:07,308] [   TRAIN] - Epoch=3/3, Step=140/282 loss=0.0597 acc=0.9875 lr=0.000050 step/sec=10.78 | ETA 00:01:37
    [2021-11-15 15:11:08,232] [   TRAIN] - Epoch=3/3, Step=150/282 loss=0.0401 acc=0.9844 lr=0.000050 step/sec=10.83 | ETA 00:01:37
    [2021-11-15 15:11:09,155] [   TRAIN] - Epoch=3/3, Step=160/282 loss=0.0392 acc=0.9844 lr=0.000050 step/sec=10.84 | ETA 00:01:37
    [2021-11-15 15:11:10,074] [   TRAIN] - Epoch=3/3, Step=170/282 loss=0.0265 acc=0.9938 lr=0.000050 step/sec=10.88 | ETA 00:01:36
    [2021-11-15 15:11:10,995] [   TRAIN] - Epoch=3/3, Step=180/282 loss=0.0676 acc=0.9844 lr=0.000050 step/sec=10.85 | ETA 00:01:36
    [2021-11-15 15:11:11,921] [   TRAIN] - Epoch=3/3, Step=190/282 loss=0.0715 acc=0.9812 lr=0.000050 step/sec=10.80 | ETA 00:01:36
    [2021-11-15 15:11:12,842] [   TRAIN] - Epoch=3/3, Step=200/282 loss=0.0495 acc=0.9875 lr=0.000050 step/sec=10.85 | ETA 00:01:36
    [2021-11-15 15:11:13,767] [   TRAIN] - Epoch=3/3, Step=210/282 loss=0.0252 acc=0.9969 lr=0.000050 step/sec=10.82 | ETA 00:01:35
    [2021-11-15 15:11:14,691] [   TRAIN] - Epoch=3/3, Step=220/282 loss=0.0884 acc=0.9656 lr=0.000050 step/sec=10.82 | ETA 00:01:35
    [2021-11-15 15:11:15,613] [   TRAIN] - Epoch=3/3, Step=230/282 loss=0.0596 acc=0.9812 lr=0.000050 step/sec=10.85 | ETA 00:01:35
    [2021-11-15 15:11:16,546] [   TRAIN] - Epoch=3/3, Step=240/282 loss=0.0949 acc=0.9656 lr=0.000050 step/sec=10.72 | ETA 00:01:35
    [2021-11-15 15:11:17,473] [   TRAIN] - Epoch=3/3, Step=250/282 loss=0.0492 acc=0.9875 lr=0.000050 step/sec=10.78 | ETA 00:01:35
    [2021-11-15 15:11:18,400] [   TRAIN] - Epoch=3/3, Step=260/282 loss=0.0410 acc=0.9844 lr=0.000050 step/sec=10.79 | ETA 00:01:34
    [2021-11-15 15:11:19,327] [   TRAIN] - Epoch=3/3, Step=270/282 loss=0.1309 acc=0.9563 lr=0.000050 step/sec=10.78 | ETA 00:01:34
    [2021-11-15 15:11:20,242] [   TRAIN] - Epoch=3/3, Step=280/282 loss=0.1014 acc=0.9719 lr=0.000050 step/sec=10.93 | ETA 00:01:34
    [2021-11-15 15:11:21,865] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - [Evaluation result] avg_acc=0.8893
    [2021-11-15 15:11:21,867] [    INFO] - Saving model checkpoint to ./ckpt/epoch_3



```python
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
```

    INFO:filelock:Lock 140344245123664 acquired on /home/aistudio/.paddlehub/tmp/ernie


    Download https://bj.bcebos.com/paddlehub/paddlehub_dev/ernie_2.0.2.tar.gz
    [##################################################] 100.00%
    Decompress /home/aistudio/.paddlehub/tmp/tmpxlci7_sa/ernie_2.0.2.tar.gz
    [##################################################] 100.00%


    [2021-11-15 15:14:15,129] [    INFO] - Successfully installed ernie-2.0.2
    INFO:filelock:Lock 140344245123664 released on /home/aistudio/.paddlehub/tmp/ernie
    [2021-11-15 15:14:15,133] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-1.0
    [2021-11-15 15:14:15,135] [    INFO] - Downloading ernie_v1_chn_base.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams
    100%|██████████| 392507/392507 [00:14<00:00, 27622.18it/s]
    [2021-11-15 15:14:32,016] [    INFO] - Loaded parameters from /home/aistudio/data/data16287/ckpt/best_model/model.pdparams
    [2021-11-15 15:14:32,024] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt and saved to /home/aistudio/.paddlenlp/models/ernie-1.0
    [2021-11-15 15:14:32,032] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt
    100%|██████████| 90/90 [00:00<00:00, 2964.24it/s]


    Data: 昌平京基鹭府10月29日推别墅1200万套起享97折  新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。  京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。 	 Lable: 教育
    Data: 尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。  这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。  片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。 	 Lable: 家居
    Data: 罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪  记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。  拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。 	 Lable: 教育
    Data: 新总督致力提高加拿大公立教育质量  滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。  约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。 	 Lable: 家居

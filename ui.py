from bokeh.core.properties import value
from bokeh.io import curdoc, show
from bokeh.layouts import layout, Spacer
from bokeh.layouts import widgetbox as wb
from bokeh.models import ColumnDataSource
from bokeh.models import Spinner
from bokeh.models import widgets as wd
from bokeh.models.annotations import Label, Title
from bokeh.models.glyphs import Step, Text, VBar
from bokeh.plotting import figure
from bs4 import BeautifulSoup as bs
import random
import os
import yaml

## predefs
button_name =  ["1", "2", "3"]
model_list = ["YOLOv3", "FasterRNN", "ETC"]
backbone_list = ["ResNet18", "ResNet50", "MobileNetv1"]
yn_list = ["yes", "no"]
premodel_list = ["COCO", "YOLO", "IMAGENET"]
dataset_template_type = ["COCO", "VOC"]

configs = {}
model_configs = {}

source = ""


## widgets
page_title = wd.Div(text = '<b><font size="6">联邦学习工具</font></b>')

fed_config_p = wd.Div(text = '<b><font size="4">联邦学习设置</font></b>') 
pro_wtt = wd.Spinner(title="最大等待用户相应时间", low=1 ,high=20,step=1,value=5)
worker_num = wd.Spinner(title="调用用户", low=1 ,high=10,step=1,value=2)

model_select_p = wd.Div(text = '<b><font size="4">模型设置</font></b>')
model_select = wd.Select(title="模型选择", value="YOLOv3", options=model_list)
backbone_select = wd.Select(title="Backbone选择", value="MobileNetv1", options=backbone_list)
useFPN_p = wd.Paragraph(text="使用FPN")
useFPN_radio = wd.RadioGroup(labels=yn_list)
custpre_p = wd.Paragraph(text="使用自定义预训练模型")
custpre_radio = wd.RadioGroup(labels=yn_list)
pretrainmodel_select = wd.Select(title="预训练模型选择", value="IMAGENET", options=premodel_list)
useGPU_p = wd.Paragraph(text="使用GPU")
useGPU_radio = wd.RadioGroup(labels=yn_list)

model_parameter_p = wd.Div(text='<b><font size="4">模型参数</font></b>')
input_size_p = wd.Paragraph(text = "图像输入尺寸")
input_size_min = wd.TextInput(title = "min(px)") #格式修改
input_size_max = wd.TextInput(title = "max(px)")

train_parameter_p = wd.Div(text='<b><font size="4">训练参数</font></b>')
epoch_spinner = Spinner(title="迭代轮数（Epoch）", low=1, high=9999, step=1, value=12)
lr_spinner = Spinner(title="学习率（Learning Rate）", low=0, high=1, step=0.000001, value=0.00125)
batch_size = Spinner(title="批大小（Batch Size）", low=1, high=9999, step=1, value=1)
max_iter_per_epoch = Spinner(title="最大迭代次数", low=1, high=100000, step=1000, value=20000)
# num_classes

optimize_opt_p = wd.Div(text='<b><font size="4">优化策略</font></b>')
data_enhance_p = wd.Paragraph(text="数据增强")
data_enhance_radio = wd.RadioGroup(labels=yn_list)

terminal_info_p = wd.Div(text = '<b><font size="4">终端信息</font></b>')
terminal_info_display = wd.Div(text = "训练设置已保存<br>数据集加载完成<br>FedVision启动")

button_group = wd.RadioButtonGroup(labels=button_name, active = 1)

input_doc_location = wd.TextInput(title = "输入测试集相对路径")
refreshButton = wd.Button(label="启动", width=100)
save_button = wd.Button(label="存储", width = 100)

template_mode_select = wd.Select(title = "数据集格式选择", value = "VOC", options = dataset_template_type)

size_of_train_set = wd.TextInput(title = "训练集大小")
size_of_test_set = wd.TextInput(title = "测试集大小")
size_of_vali_set = wd.TextInput(title = "验证集大小")

trainset_config_save = wd.Button(label = "存储数据集设置")

figure_refresh_button = wd.Button(label = "刷新图表")

dataset_text = wd.Div(text = "")

empty_block = wd.Paragraph(text = " ")

logo = wd.Div(text = '<img src="https://z3.ax1x.com/2021/07/22/WwXPUS.png" height="100px" /> ')
lgu_logo = wd.Div(text = '<img src="https://z3.ax1x.com/2021/08/08/flBGUs.png" height="100px" />')


all_list = []
test_list = []
val_list = []


dataset_info = figure(title="目标类型及数量统计")
loss_info = figure(title="Loss")
bbox_info = figure(title="Bbox_map")
loss = [1293.336914, 27.080219, 24.2502, 24.676718, 22.545147, 23.334381, 21.437468, 22.315378, 19.885063, 19.462503, 20.317566, 20.580782, 19.529863, 20.537045, 18.119638, 17.980953, 19.674805, 18.848146, 17.512794, 18.509291, 17.067566, 18.913307, 18.251993, 17.501133, 17.61598, 17.383341, 16.522991, 16.678352, 15.938581, 17.0481, 16.57539, 17.000517, 15.833963, 15.980446, 15.474256, 16.68627, 16.537689, 17.164871, 15.669952, 16.675564, 15.851392, 16.496122, 15.596779, 15.333354, 16.012575, 16.978878, 15.37412, 15.709514, 15.047617, 16.157396, 15.311027, 16.211874, 16.598406, 15.604307, 14.625225, 14.985044, 14.195051, 15.243348, 14.86463, 14.854979]
bbox_map = [0.634667, 0.921526, 2.533716, 1.744398, 2.604492, 1.334795, 10.43986, 4.518972, 4.026264, 18.51807, 8.877012, 7.658616, 5.620845, 12.14614, 14.7235, 20.81075, 14.98679, 18.42379, 14.25338, 12.20238, 23.44585, 24.04879, 33.54739, 26.63958, 25.88476, 21.21089, 29.22838, 32.03198, 30.10481, 34.90458, 36.73026, 30.65362, 24.94199, 36.88293, 35.77115, 39.78789, 44.96787, 42.57825, 36.53271, 47.42652, 48.91905, 53.22578, 51.88363, 47.96305, 54.83479, 51.70324, 44.40368, 51.50521, 55.72616, 54.17528, 57.30561, 53.20639, 49.04498, 57.68252, 49.71831, 50.45835, 45.05687, 53.70412, 48.40931, 56.40606]
loss_info.yaxis.axis_label = "Loss"
bbox_info.yaxis.axis_label = "Bbox_map"
loss_info.xaxis.axis_label = "Epoch"
bbox_info.xaxis.axis_label = "Epoch"
rangex_loss = [i for i in range(0,60)]
rangex_bbox = [i for i in range(0,60)]
data_loss_source = ColumnDataSource(data = {
    "Loss" : loss,
    "Epoch": rangex_loss
})

loss_info.line("Epoch", "Loss", source = data_loss_source)
bbox_info.line(x = rangex_bbox, y = bbox_map)


## layout
test = layout([
    [wb(logo, width=600), wb(lgu_logo,  width=200)],
    [wb(page_title)],
    [wb(empty_block)],
    [wb(fed_config_p)],
    [wb(pro_wtt)],
    [wb(worker_num)],
    [wb(empty_block)],
    [wb(model_select_p)],
    [wb(model_select)],
    [wb(backbone_select)],
    [wb(useFPN_p, width = 100), wb(useFPN_radio)],
    [wb(custpre_p, width = 100), wb(custpre_radio)],
    [wb(pretrainmodel_select)],
    [wb(empty_block)],
    [wb(model_parameter_p)],
    [wb(input_size_p, width=150), wb(input_size_min), wb(input_size_max)],
    [wb(useGPU_p, width = 100), wb(useGPU_radio)],
    [wb(empty_block)],
    [wb(train_parameter_p)],
    [wb(epoch_spinner)],
    [wb(lr_spinner)],
    [wb(batch_size)],
    [wb(max_iter_per_epoch)],
    [wb(save_button)],
    [wb(refreshButton)],
    [wb(empty_block)],
    [wb(terminal_info_p)],
    [wb(terminal_info_display)]
    ])
test2 = layout([[wb(input_doc_location)],
                [wb(template_mode_select)],
                [wb(refreshButton, width = 200)],
                [wb(size_of_test_set)],
                [wb(size_of_train_set)],
                [wb(size_of_vali_set)],
                [wb(trainset_config_save)],
                [wb(dataset_text)],
                [wb(dataset_info)]
                ])
test3 = layout([[wb(loss_info), wb(figure_refresh_button)],
                [wb(bbox_info)]])

## functions                
# 读取数据集文件，获取统计信息
def load_info():
    global source 
    source = input_doc_location.value
    listdir = os.listdir(source + "/Annotations")
    itemlist = []
    pointdict = {}
    cr_text = ""
    key_set = []
    value_set = []
    for item in listdir:
        if (item.endswith(".xml")):
            itemlist.append(item)
    print(len(itemlist))
    cr_text += "数据集含有"+ str(len(itemlist))+ "张图片。<br>"
    cr_text += "目标类型：<br>"
    for item in itemlist:
        with open(source + "/Annotations/" + item, "r", encoding='utf-8') as fh:
            text = fh.read()
            soup = bs(text, "xml")
            for obj in soup.find_all("object"):
                ftype = obj.find("name").contents[0]
                if ftype in pointdict:
                    pointdict[ftype] += 1
                else:
                    pointdict[ftype] = 1   
    for key, value in pointdict.items():
        cr_text += str(key) + " " + str(value) + "<br>"
        key_set.append(key)
        value_set.append(value)
    dataset_text.text = cr_text
    print(value_set)
    dataset_info.vbar(x=[0, 1, 2], top=value_set, width=0.5, bottom=0)

# 保存数据集分割
def save_dataset_split():
    all_list = load_pics()
    random_select(all_list, int(size_of_train_set.value))

# 读取图片文件文件名
def load_pics():
    empty = []
    pic_dir = os.listdir(source + "/JPEGImages")
    for item in pic_dir:
        if(item.endswith(".jpg")):
            empty.append(item)
    return empty

# 对数据集按照配置进行随机分割
# all: 文件名数组 split：分割
def random_select(all, split):
    train_list = random.sample(all, split) #train_val
    val_list = set(all).difference(set(train_list))
    train_file = open(source + "/train.txt", "w", encoding='utf-8')
    for item in list(train_list):
        xml_item = item.split(".jpg")[0]
        xml_item += ".xml"
        full_item = source + "/JPEGImages/" + item
        full_xml_item = source + "/Annotations/" + xml_item
        train_file.write(full_item + " " + full_xml_item + "\n")
    train_file.close()
    val_file = open(source + "/val.txt", "w", encoding='utf-8')
    for item in list(val_list):
        xml_item = item.split(".jpg")[0]
        xml_item += ".xml"
        full_item = source + "/JPEGImages/" + item
        full_xml_item = source + "/Annotations/" + xml_item
        val_file.write(full_item + " " + full_xml_item + "\n")
    val_file.close()

# 读取全局设置
def load_configs(name = "config_with_comments.yaml"):
    file = open(name, 'r', encoding='utf-8')
    file_data = file.read()
    config_dict = yaml.load(file_data, Loader=yaml.BaseLoader)
    print(config_dict)
    file.close()
    return config_dict

# 读取模型设置
def load_model_configs(name = "withcomment1.yml"):
    file = open(name, 'r', encoding='utf-8')
    file_data = file.read()
    model_dict = yaml.load(file_data, Loader=yaml.BaseLoader)
    print(model_dict)
    file.close()
    return model_dict

# 写入配置至文件
# file_name: 输出文件名 dict: 配置保存的dict
def write_back(file_name, dict):
    file = open(file_name, 'w', encoding="utf-8")
    yaml.dump(dict, file)
    file.close()

def set_prop_wtt(value, target_dict):
    target_dict["job_config"]["proposal_wait_time"] = value

def set_worker_num(value, target_dict):
    target_dict["job_config"]["worker_num"] = value

def set_epoch(value, target_dict):
    target_dict["job_config"]["max_iter"] = value

def set_device_config(value, target_dict):
    target_dict["job_config"]["device"] = "gpu" if value else "cpu"

def set_max_iter(value, target_dict):
    target_dict["max_iters"] = value

def set_learning_rate(value, target_dict):
    target_dict["LearningRate"]["base_lr"] = value

def set_image_shape(value, target_dict):
    target_dict["TrainReader"]["inputs_def"]["image_shape"][1] = value
    target_dict["TrainReader"]["inputs_def"]["image_shape"][2] = value

def set_batch_size(value, target_dict):
    target_dict["TrainReader"]["batch_size"] = value
    target_dict["EvalReader"]["batch_size"] = value
    target_dict["TestReader"]["batch_size"] = value

def set_dataset_dir(target_dict):
    target_dict["TrainReader"]["dataset"]["dataset_dir"] = source
    target_dict["EvalReader"]["dataset"]["dataset_dir"]= source

def set_anno_path(target_dict):
    target_dict["TrainReader"]["dataset"]["anno_path"] = source + "/train.txt"
    target_dict["EvalReader"]["dataset"]["anno_path"]= source + "/val.txt"

def wb_to_yml():
    model_config_name = "model_configs.yml"
    config_name = "configs.yml"
    m_config = open(model_config_name, 'w', encoding='utf-8') 
    yaml.safe_dump(model_configs, m_config, default_style=None)
    m_config.close()
    config = open(config_name, 'w', encoding='utf-8')
    yaml.safe_dump(configs, config, default_style=None)
    config.close()
    
# 读取trainer的日志文件
# path: trainer的路径
def load_trainer_log(path):
    log_file = open(path, "r")
    line = log_file.readline()
    array = []
    while line:
        if (line.find("array") != -1):
            res = line.split("[")[2].split("]")[0]
            array.append(float(res))
        line = log_file.readline()
    log_file.close()
    return array

# 测试用
def sort_directory():
    list = ["master1-20211019202024-1", "master1-20211020002839-1",  "master1-20211020010003-4",
        "master1-20211019233547-2", "master1-20211020003417-2",
        "master1-20211019233834-3",  "master1-20211020004934-3"]
    list.sort()
    print(list)


# 保存UI中配置到文件并启动训练
def set_configs_all():
    terminal_list = [
            "cd /data/projects/fedvision && source venv/bin/activate && export PYTHONPATH=$PYTHONPATH:/data/projects/fedvision/FedVision && sh FedVision/examples/paddle_detection/run.sh 127.0.0.1:10002"
    ]
    set_prop_wtt(pro_wtt.value, configs)
    set_worker_num(worker_num.value, configs)
    set_epoch(epoch_spinner.value, configs)
    set_device_config(useGPU_radio.active, configs)
    set_max_iter(max_iter_per_epoch.value, model_configs)
    set_learning_rate(lr_spinner.value, model_configs)
    set_image_shape(input_size_max.value_input, model_configs)
    set_batch_size(batch_size.value, model_configs)
    set_dataset_dir(model_configs)
    set_anno_path(model_configs)
    wb_to_yml()   
    for state in terminal_list:
        if os.system(state) == 0:
            print(state, " success")

# 刷新图表文件
def refresh_figure_data():
    dir = "/data/projects/fedvision/FedVision/logs/jobs"
    dir_list = []
    for dirs in os.listdir(dir):
        dir_list.append(dirs)
    dir_list.sort()
    print(dir_list)
    print(dir_list[-1])
    final_path = dir + '/' + dir_list[-1] + "/trainer_0/trainer.log"
    print(final_path)
    data = load_trainer_log(final_path)
    range_data = [i for i in range(len(data))]
    data_loss_source.data = {
    "Loss" : data,
    "Epoch": range_data
    }

## widgets connection
refreshButton.on_click(load_info)
trainset_config_save.on_click(save_dataset_split)
save_button.on_click(set_configs_all)
figure_refresh_button.on_click(refresh_figure_data)

configs = load_configs()
model_configs = load_model_configs()
page_test = wd.Panel(child = test, title="参数设置")
page2_test = wd.Panel(child = test2, title = "数据集设置")
page3_test = wd.Panel(child = test3, title="训练结果")
tabs = wd.Tabs(tabs = [page2_test, page_test, page3_test])
curdoc().add_root(tabs)


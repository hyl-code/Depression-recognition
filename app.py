import os
import time

from flask import Flask, render_template, redirect, url_for
from flask import request
from werkzeug.utils import secure_filename
import os
import subprocess
import json

app = Flask(__name__)
UPLOAD_FOLDER = './EATD-Corpus'
ALLOWED_EXTENSIONS = {'wav', 'txt'}


def script_run(cmd):
    '''
    设置一个进程返回标记
    '''
    res_mark = '[res_json]'
    subp = subprocess.Popen(cmd, encoding='utf-8', stdout=subprocess.PIPE)
    out, err = subp.communicate()

    res = None
    for line in out.splitlines():
        if line.startswith((res_mark,)):
            res = json.loads(line.replace(res_mark, '', 1))
            break
    return res


@app.route('/', methods=['POST', 'GET'])
def homePage():
    return render_template('homepage.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    i = 1
    if request.method == 'POST':
        files = request.files.getlist("file")
        for file in files:
            filename = secure_filename(file.filename)
            if file and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS:
                folder = UPLOAD_FOLDER + '/t_' + str(i)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                file.save(os.path.join(folder, filename))
        i += 1
        return analysis()
    else:
        return render_template('upload.html')


D_list = ['one_for_audio.py', 'one_for_text.py', 'two.py']


@app.route('/analysis', methods=['POST'])
def analysis():
    for pyfile in D_list:
        if os.path.isfile(pyfile) and pyfile.endswith('.py'):
            os.system("python %s" % os.path.join(os.getcwd(), pyfile))
    cmd = 'python ./classification.py'
    res = script_run(cmd)
    print(res)
    if res == '抑郁':
        text = '1. 不管你现在多么的痛苦，都要坚持住。不要被眼前的困难所打败，如果很累了，那就降低要求，回家好好休息一段时间，给自己一段时间来疗愈心情或者寻求医生正规地治疗。<br/>' \
               '2. 科学治疗，对症治疗，对症用药治疗了，加上心理治疗会更好。<br/>' \
               '3. 坚持运动，运动出汗，对提高情绪也是有帮助的，因为出汗大脑里会分泌一些多巴胺出来，这种神经递质能使人愉悦。<br/>' \
               '4. 吸收正能量，负面能量太多是黑暗；而阳光起来，需要靠积极的正能量。<br/>' \
               '5. 带着症状去生活，不去过度关注自己的症状，忙碌充实起来，反而症状会消失掉。'
    elif res == '正常':
        text = "1. 工作或是生活当中，都要注意建议良好的人际关系，并且在有压力的时候积极倾诉和求助。<br/>" \
               "2. 在平时生活当中，一定要寻找自己的生活乐趣，要不断的尝试创新，这样能给自己的精神上得到一定的满足，能够放松身心，起到保持心理健康的效果。<br/>" \
               "3. 时常和家人保持联系，家是我们避风的港湾，而家庭环境所具有的安全感会造成非常重要的影响，有了家人的爱护和理解我们就会有了安全感。<br/>" \
               "4. 客观的对自身进行评价，评价不宜过高过低。<br/>" \
               "5. 加强与外界的接触，可以丰富自身精神生活，亦或可以及时调整自己适应环境。"
    return render_template('analysis.html', result=res, advice=text)


@app.route('/instruction')
def instruction():
    return render_template('instruction.html')


@app.route('/consult')
def consult():
    return render_template('consult.html')


@app.route('/contact', methods=['POST', 'GET'])
def contact():
    if request.method == 'POST':
        return redirect(url_for('contact_successfully'))
    return render_template('contact.html')


@app.route('/contact_successfully')
def contact_successfully():
    return render_template('contact_successfully.html')


if __name__ == '__main__':
    app.run(debug=True)

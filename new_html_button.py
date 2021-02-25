import os
import numpy as np
from xml.dom import minidom

base_path = './result/test_nogt_aug_weighted/valid_prefix_25_result_1_2_100'
f = open(os.path.join(base_path, 'demo.html'), 'w')
score_weights = (1., 2., 100.)

size = 256
f.write('<html>\n')
for idx, name in enumerate(os.listdir(base_path)):
    if name[-4:] == 'html' or len(name) < 5:
        continue
    current_path = name
    info_ = np.load(os.path.join(base_path, name, 'config.npy'), allow_pickle=True).tolist()
    gt_overall_score = info_['gt']['score']
    gt_false_corner_num = info_['gt']['false_corner']
    gt_false_edge_num = info_['gt']['false_edge']
    gt_region_score = info_['gt']['region_score']

    #configxml = minidom.parse(os.path.join(base_path, '..', 'config.xml'))
    #score_weights = configxml.getElementsByTagName('score_weights')[0].getElementsByTagName('item')
    corner_weight = score_weights[0]#.firstChild.data
    edge_weight = score_weights[1]#.firstChild.data
    region_weight = score_weights[2]#.firstChild.data
    f.write('<p><b><big>' + '&nbsp;'*50
            + name + '&nbsp;'*50 + 'gt: ' + str(gt_overall_score) +
            '&nbsp;'*50 + str(gt_false_corner_num) + '&nbsp;'*50 + str(gt_false_edge_num) +
            '&nbsp;'*50 + str(gt_region_score) + '&nbsp;'*50 +
            'score = {}corner_score + {}edge_score + {}region_score'.format(corner_weight, edge_weight, region_weight) +
            '</big></b></p>')
    f.write('<p>')

    f.write('&nbsp;<img src="' + os.path.join(name, 'image.jpg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'iter_0_num_0.svg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred.svg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_corner.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_edge.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_region.png') + '" width="'+str(size)+'">')
    if os.path.exists(os.path.join(base_path,name,'heatmap.png')):
        f.write('<img src="' + os.path.join(name, 'heatmap.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name, 'maskrcnn.png') + '" width="'+str(size)+'">')

    f.write('</p>\n')

    f.write('<a href="javascript:;" id="best_'+ str(idx) +'">show best</a>\n')
    f.write('<span id="best_content_'+ str(idx) +'">  </span>\n')
    f.write('<script type="application/javascript">')

    img_html_text = ''
    for best_i in range(5):
        namePath = os.path.join(name, 'best_' + str(best_i))
        info_name = 'best_'+str(best_i)
        if info_name not in info_.keys():
            break
        overall_score = info_[info_name]['score']
        false_corner_num = info_[info_name]['false_corner']
        false_edge_num = info_[info_name]['false_edge']
        corner_score = info_[info_name]['corner_score']
        edge_score = info_[info_name]['edge_score']
        region_score = info_[info_name]['region_score']
        img_html_text+='<p><b><big>best '+str(best_i) + '&nbsp;'*7 + str(overall_score) + \
                       '&nbsp;'*30 + 'false corner: ' + str(false_corner_num) + '&nbsp;score: ' + str(corner_score) + \
                       '&nbsp;'*30 + 'false edge: ' + str(false_edge_num) + '&nbsp;score: ' + str(edge_score) + \
                       '&nbsp;'*30 + 'false region: 0 &nbsp;score: ' + str(region_score) + \
                       '</big></b></p>'

        img_html_text+='<img src="' + os.path.join(namePath+'.svg') + '" width="'+str(size)+'">'
        img_html_text+='<b><big>    </big></b>'
        img_html_text+='<img src="' + os.path.join(namePath+'_corner.png') + '" width="'+str(size)+'">'
        img_html_text+='<b><big>    </big></b>'
        img_html_text+='<img src="' + os.path.join(namePath+'_edge.png') + '" width="'+str(size)+'">'
        img_html_text+='<b><big>    </big></b>'
        img_html_text+='<img src="' + os.path.join(namePath+'_region.png') + '" width="'+str(size)+'">'

    text = 'var btn_'+ str(idx) +' = document.getElementById(\'best_'+ str(idx) +'\');\n' \
           'var content_'+ str(idx) + ' = document.getElementById(\'best_content_'+ str(idx) +'\');\n' \
           'var str_'+str(idx)+' = content_'+str(idx)+'.innerHTML;\n' \
           'var onOff_'+str(idx)+' = true;\n' \
           'btn_'+str(idx)+'.onclick = function() {\n' \
           'if(onOff_'+str(idx)+') {\n' \
           'content_'+str(idx)+'.innerHTML = \'' + img_html_text + '\';\n' \
           'btn_'+str(idx)+'.innerHTML = \'close\'\n' \
           '} else { content_'+str(idx)+'.innerHTML = str_'+str(idx)+'\nbtn_'+str(idx)+'.innerHTML=\'show best\';}\n' \
           'onOff_'+str(idx)+' = !onOff_'+str(idx)+'; return false;}'
    f.write(text)
    f.write('</script>\n')


    for _iter in range(15):
        if 'iter_{}_num_0'.format(_iter) not in info_.keys():
            break
        f.write('&nbsp;'*5)
        f.write('<a href="javascript:;" id="example_'+ str(idx) + '_iter_' + str(_iter) +'">iter '+str(_iter)+' </a>\n')
        f.write('<span id="example_content_'+ str(idx) +'_iter_' + str(_iter) +'">  </span>\n')
        f.write('<script type="application/javascript">')
        img_html_text = ''
        for _id in range(10):
            namePath = os.path.join(name, 'iter_' + str(_iter) + '_num_'+str(_id))
            info_name = 'iter_'+str(_iter)+'_num_'+str(_id)
            if info_name not in info_.keys():
                continue
            overall_score = info_[info_name]['score']
            false_corner_num = info_[info_name]['false_corner']
            false_edge_num = info_[info_name]['false_edge']
            corner_score = info_[info_name]['corner_score']
            edge_score = info_[info_name]['edge_score']
            region_score = info_[info_name]['region_score']
            img_html_text+='<p><b><big>iter '+str(_iter) + ' num ' + str(_id) + '&nbsp;'*5 + str(overall_score) + \
                           '&nbsp;'*30 + 'false corner: ' + str(false_corner_num) + '&nbsp;score: ' + str(corner_score) + \
                           '&nbsp;'*30 + 'false edge: ' + str(false_edge_num) + '&nbsp;score: ' + str(edge_score) + \
                           '&nbsp;'*30 + 'false region: 0 &nbsp;score: ' + str(region_score) + \
                           '</big></b></p>'

            img_html_text+='<img src="' + os.path.join(namePath+'.svg') + '" width="'+str(size)+'">'
            img_html_text+='<b><big>    </big></b>'
            img_html_text+='<img src="' + os.path.join(namePath+'_corner.png') + '" width="'+str(size)+'">'
            img_html_text+='<b><big>    </big></b>'
            img_html_text+='<img src="' + os.path.join(namePath+'_edge.png') + '" width="'+str(size)+'">'
            img_html_text+='<b><big>    </big></b>'
            img_html_text+='<img src="' + os.path.join(namePath+'_region.png') + '" width="'+str(size)+'">'
            img_html_text+='<b><big>    </big></b>'
            if os.path.exists(os.path.join(base_path,name,'edge_heatmap.png')):
                img_html_text+='<img src="' + os.path.join(current_path, 'edge_heatmap.png') + '" width="'+str(size)+'">'
                img_html_text+='<b><big>    </big></b>'
                img_html_text+='<img src="' + os.path.join(current_path, 'maskrcnn.png') + '" width="'+str(size)+'">'
                img_html_text+='<b><big>    </big></b>'

        text ='var btn_'+ str(idx)+'_'+str(_iter) +' = document.getElementById(\'example_'+ str(idx) + '_iter_' + str(_iter) +'\');\n' \
              'var content_'+ str(idx)+'_'+str(_iter) + \
              ' = document.getElementById(\'example_content_'+ str(idx)+'_iter_'+str(_iter) +'\');\n' \
              'var str_'+str(idx)+'_'+str(_iter)+' = content_'+str(idx)+'_'+str(_iter)+'.innerHTML;\n' \
              'var onOff_'+str(idx)+'_'+str(_iter)+' = true;\n' \
              'btn_'+str(idx)+'_'+str(_iter)+'.onclick = function() {\n' \
              'if(onOff_'+str(idx)+'_'+str(_iter)+') {\n' \
              'content_'+str(idx)+'_'+str(_iter)+'.innerHTML = \'' + img_html_text + '\';\n' \
              'btn_'+str(idx)+'_'+str(_iter)+'.innerHTML = \'close\'\n' \
              '} else { ' \
              'content_'+str(idx)+'_'+str(_iter)+'.innerHTML = ' \
              'str_'+str(idx)+'_'+str(_iter)+'\nbtn_'+str(idx)+'_'+str(_iter)+'.innerHTML=\'iter '+str(_iter)+'\';}\n' \
              'onOff_'+str(idx)+'_'+str(_iter)+' = !onOff_'+str(idx)+'_'+str(_iter)+'; return false;}'
        f.write(text)
        f.write('</script>\n')


    #if idx == 50:
    #    break

f.write('</html>')

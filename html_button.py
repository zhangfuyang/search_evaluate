import os
import numpy as np

base_path = '/local-scratch/fuyang/result/beam_search_new/corner_edge_region_threading_region_60/'
img_path = '/local-scratch/fuyang/cities_dataset'
entire_mask_path = '/local-scratch/fuyang/result/corner_edge_region/entire_region_mask'
f = open(os.path.join(base_path, 'demo.html'), 'w')

size = 256
f.write('<html>\n')
for idx, name in enumerate(os.listdir(base_path)):
    if name[-4:] == 'html' or len(name) < 5:
        continue
    current_path = name
    info_f = open(os.path.join(base_path, name, 'gt_pred_info.txt'))
    gt_overall_score = info_f.readline()[:-1]
    gt_false_corner_num = info_f.readline()[:-1]
    gt_false_edge_num = info_f.readline()[:-1]
    gt_false_region_num = info_f.readline()
    f.write('<p><b><big>' + '&nbsp;'*50
            + name + '&nbsp;'*50 + 'gt: ' + str(gt_overall_score) +
            '&nbsp;'*50 + str(gt_false_corner_num) + '&nbsp;'*50 + str(gt_false_edge_num) +
            '&nbsp;'*50 + str(gt_false_region_num) + '&nbsp;'*50 +
            'score = corner_score + 2edge_score + 60region_score' + '</big></b></p>')
    f.write('<p>')
    if os.path.exists(os.path.join(base_path,name,'edge_heatmap.png')):
        f.write('<img src="' + os.path.join(name, 'edge_heatmap.png') + '" width="'+str(size)+'">')
        f.write('<img src="' + os.path.join(name, 'maskrcnn.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(img_path, 'rgb', name+'.jpg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'iter_0_num_0.svg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred.svg') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_corner.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_edge.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(name,'gt_pred_region.png') + '" width="'+str(size)+'">')
    f.write('&nbsp;<img src="' + os.path.join(entire_mask_path, name+'.png') + '" width="'+str(size)+'">')

    f.write('</p>\n')

    f.write('<a href="javascript:;" id="best_'+ str(idx) +'">show best</a>\n')
    f.write('<span id="best_content_'+ str(idx) +'">  </span>\n')
    f.write('<script type="application/javascript">')

    img_html_text = ''
    for best_i in range(5):
        namePath = os.path.join(name, 'best_' + str(best_i))
        if not os.path.exists(os.path.join(base_path,namePath+'_info.txt')):
            break
        info_f = open(os.path.join(base_path, namePath+'_info.txt'))
        overall_score = info_f.readline()[:-1]
        false_corner_num = info_f.readline()[:-1]
        false_edge_num = info_f.readline()[:-1]
        false_region_num = info_f.readline()[:-1]
        corner_score = info_f.readline()[:-1]
        edge_score = info_f.readline()[:-1]
        region_score = info_f.readline()[:-1]
        img_html_text+='<p><b><big>best '+str(best_i) + '&nbsp;'*7 + overall_score + \
                       '&nbsp;'*30 + 'false corner: ' + false_corner_num + '&nbsp;score: ' + corner_score +\
                       '&nbsp;'*30 + 'false edge: ' + false_edge_num + '&nbsp;score: ' + edge_score + \
                       '&nbsp;'*30 + 'false region: ' + false_region_num + '&nbsp;score: ' + region_score + \
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


    for _iter in range(6):
        f.write('&nbsp;'*5)
        f.write('<a href="javascript:;" id="example_'+ str(idx) + '_iter_' + str(_iter) +'">iter '+str(_iter)+' </a>\n')
        f.write('<span id="example_content_'+ str(idx) +'_iter_' + str(_iter) +'">  </span>\n')
        f.write('<script type="application/javascript">')
        temp_count = 0
        img_html_text = ''
        for _id in range(3,-1,-1):
            namePath = os.path.join(name, 'iter_' + str(_iter) + '_num_'+str(_id))
            if not os.path.exists(os.path.join(base_path,namePath+'_info.txt')):
                continue
            temp_count += 1
            if temp_count == 3:
                break
            info_f = open(os.path.join(base_path, namePath+'_info.txt'))
            overall_score = info_f.readline()[:-1]
            false_corner_num = info_f.readline()[:-1]
            false_edge_num = info_f.readline()[:-1]
            false_region_num = info_f.readline()[:-1]
            corner_score = info_f.readline()[:-1]
            edge_score = info_f.readline()[:-1]
            region_score = info_f.readline()[:-1]
            img_html_text+='<p><b><big>iter '+str(_iter) + ' num ' + str(_id) + '&nbsp;'*5 + overall_score + \
                           '&nbsp;'*30 + 'false corner: ' + false_corner_num + '&nbsp;score: ' + corner_score + \
                           '&nbsp;'*30 + 'false edge: ' + false_edge_num + '&nbsp;score: ' + edge_score + \
                           '&nbsp;'*30 + 'false region: ' + false_region_num + '&nbsp;score: ' + region_score + \
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
              'var content_'+ str(idx)+'_'+str(_iter) + ' = document.getElementById(\'example_content_'+ str(idx)+'_iter_'+str(_iter) +'\');\n' \
              'var str_'+str(idx)+'_'+str(_iter)+' = content_'+str(idx)+'_'+str(_iter)+'.innerHTML;\n' \
              'var onOff_'+str(idx)+'_'+str(_iter)+' = true;\n' \
              'btn_'+str(idx)+'_'+str(_iter)+'.onclick = function() {\n' \
              'if(onOff_'+str(idx)+'_'+str(_iter)+') {\n' \
              'content_'+str(idx)+'_'+str(_iter)+'.innerHTML = \'' + img_html_text + '\';\n' \
              'btn_'+str(idx)+'_'+str(_iter)+'.innerHTML = \'close\'\n' \
              '} else { content_'+str(idx)+'_'+str(_iter)+'.innerHTML = str_'+str(idx)+'_'+str(_iter)+'\nbtn_'+str(idx)+'_'+str(_iter)+'.innerHTML=\'iter '+str(_iter)+'\';}\n' \
              'onOff_'+str(idx)+'_'+str(_iter)+' = !onOff_'+str(idx)+'_'+str(_iter)+'; return false;}'
        f.write(text)
        f.write('</script>\n')


    if idx == 50:
        break

f.write('</html>')

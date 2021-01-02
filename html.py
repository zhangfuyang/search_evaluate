import os

base_path = '/local-scratch/fuyang/result/beam_search_add_edge/heatmap/'
f = open(os.path.join(base_path, 'demo.html'), 'w')

size = 256
f.write('<html>\n')
for idx, name in enumerate(os.listdir(base_path)):
    if name[-4:] == 'html' or len(name) < 5:
        continue
    current_path = name
    f.write('<p><b><big>'+ '&nbsp;'*15 +'edge_heatmap' + '&nbsp;'*30 + 'maskrcnn' + '&nbsp;'*30
            + name + '&nbsp;'*90 + 'score = corner_score + 2edge_score + 4region_score' + '</big></b></p>')
    f.write('<p>')
    f.write('<img src="' + os.path.join(name, 'edge_heatmap.png') + '" width="'+str(size)+'">')
    f.write('<img src="' + os.path.join(name, 'maskrcnn.png') + '" width="'+str(size)+'">')
    f.write('<img src="' + os.path.join('rgb', name+'.jpg') + '" width="'+str(size)+'">')
    f.write('</p>')


    for best_i in range(5):
        namePath = os.path.join(name, 'best_' + str(best_i))
        if not os.path.exists(os.path.join(base_path,namePath+'_info.txt')):
            break
        info_f = open(os.path.join(base_path, namePath+'_info.txt'))
        overall_score = info_f.readline()[:-1]
        false_corner_num = info_f.readline()[:-1]
        false_edge_num = info_f.readline()[:-1]
        false_region_num = info_f.readline()
        f.write('<p><b><big>best '+str(best_i) + '&nbsp;'*7 + overall_score +
                '&nbsp;'*35 + 'false corner: ' + false_corner_num +
                '&nbsp;'*35 + 'false edge: ' + false_edge_num +
                '&nbsp;'*35 + 'false region: ' + false_region_num +
                '</big></b></p>')

        f.write('<img src="' + os.path.join(namePath+'.svg') + '" width="'+str(size)+'">')
        f.write('<b><big>    </big></b>')
        f.write('<img src="' + os.path.join(namePath+'_corner.png') + '" width="'+str(size)+'">')
        f.write('<b><big>    </big></b>')
        f.write('<img src="' + os.path.join(namePath+'_edge.png') + '" width="'+str(size)+'">')
        f.write('<b><big>    </big></b>')
        f.write('<img src="' + os.path.join(namePath+'_region.png') + '" width="'+str(size)+'">')
        f.write('<b><big>    </big></b>')
        f.write('<img src="' + os.path.join('rgb', name+'.jpg') + '" width="'+str(size)+'">')

    for _iter in range(6):
        temp_count = 0
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
            false_region_num = info_f.readline()
            f.write('<p><b><big>iter '+str(_iter) + ' num ' + str(_id) + '&nbsp;'*5 + overall_score +
                    '&nbsp;'*35 + 'false corner: ' + false_corner_num +
                    '&nbsp;'*35 + 'false edge: ' + false_edge_num +
                    '&nbsp;'*35 + 'false region: ' + false_region_num +
                    '</big></b></p>')

            f.write('<img src="' + os.path.join(namePath+'.svg') + '" width="'+str(size)+'">')
            f.write('<b><big>    </big></b>')
            f.write('<img src="' + os.path.join(namePath+'_corner.png') + '" width="'+str(size)+'">')
            f.write('<b><big>    </big></b>')
            f.write('<img src="' + os.path.join(namePath+'_edge.png') + '" width="'+str(size)+'">')
            f.write('<b><big>    </big></b>')
            f.write('<img src="' + os.path.join(namePath+'_region.png') + '" width="'+str(size)+'">')
            f.write('<b><big>    </big></b>')
            if os.path.exists(os.path.join(base_path,name,'edge_heatmap.png')):
                f.write('<img src="' + os.path.join(current_path, 'edge_heatmap.png') + '" width="'+str(size)+'">')
                f.write('<b><big>    </big></b>')
                f.write('<img src="' + os.path.join(current_path, 'maskrcnn.png') + '" width="'+str(size)+'">')
                f.write('<b><big>    </big></b>')
            f.write('<img src="' + os.path.join('rgb', name+'.jpg') + '" width="'+str(size)+'">')

    if idx == 50:
        break

f.write('</html>')
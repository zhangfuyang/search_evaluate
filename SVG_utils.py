import svgwrite
import random
import colorsys


def random_colors(N, bright=True, same=False, colors=None):
    brightness = 1.0 if bright else 0.7
    if colors is None or same:
        if same:
            hsv = [(0, 1, brightness) for i in range(N)]
        else:
            hsv = [(i / N, 1, brightness) for i in range(N)]
    else:
        hsv = [(colors[i], 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def svg_generate(corners, edges, name, samecolor=False, colors=None, image_link=None):
    dwg = svgwrite.Drawing(name+'.svg', size=(u'256', u'256'))
    shapes = dwg.add(dwg.g(id='shape', fill='black'))
    colors = random_colors(edges.shape[0], same=samecolor, colors=colors)
    if image_link is not None:
        shapes.add(dwg.image(href=image_link, size=(256, 256)))
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i,0]
        b = edges[edge_i,1]
        if samecolor:
            shapes.add(dwg.line((int(corners[a,1]), int(corners[a,0])), (int(corners[b,1]), int(corners[b,0])),
                                stroke='red', stroke_width=1, opacity=0.8))
        else:
            shapes.add(dwg.line((int(corners[a,1]), int(corners[a,0])), (int(corners[b,1]), int(corners[b,0])),
                                stroke=svgwrite.rgb(colors[edge_i][0] * 255, colors[edge_i][1] * 255, colors[edge_i][2] * 255, '%'),
                                stroke_width=2))
    for i in range(corners.shape[0]):
        shapes.add(dwg.circle((int(corners[i][1]), int(corners[i][0])), r=2,
                                  stroke='green', fill='white', stroke_width=1, opacity=0.8))
    return dwg

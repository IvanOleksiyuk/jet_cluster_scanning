import sys
from PIL import Image

images_names=["plots2/chi_s0none_>5sigma.png",
              "plots2/chi_s0none_kmeans_der.png",
              "plots2/chi_s1sqrt_>5sigma.png",
              "plots2/chi_s1sqrt_kmeans_der.png"]

images = [Image.open(x) for x in images_names]
width, height = images[0].size

a=0#280
new_im = Image.new('RGB', (width*2, height*2))

new_im.paste(images[0], (0,0))
new_im.paste(images[1], (width,0))
new_im.paste(images[2], (0,height))
new_im.paste(images[3], ((width-a),height))


new_im.save('plots2/chi_all1.png')
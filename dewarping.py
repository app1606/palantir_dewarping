from PIL import Image, ImageDraw
from scipy import ndimage, interpolate
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm

est_points_num = 1000

def dewarp(path, top_points, bottom_points):
    
    def poly(points):
        xs = list(map(lambda x: x[0], points))
        ys = list(map(lambda x: x[1], points))
        poly_full = interpolate.lagrange(xs,ys)

        return poly_full 

    def cheb_points(n, points):
        return [(points[0][0] + points[-1][0]) / 2 +
                (points[-1][0] - points[0][0]) / 2 * np.cos((2 * k - 1) * np.pi/(2 * n)) for k in range(1, n + 1)]

    def line(n, points):
        cheb = cheb_points(n, points)

        return list(zip(cheb , poly(points)(cheb)))
    
    
    
    img = Image.open(path)
    
    
    top_line = line(est_points_num, top_points)
    bottom_line = line(est_points_num, bottom_points)
    
    top_left = np.array((top_line[-1])) #calculating edges 
    top_right = np.array((top_line[0]))
    bottom_left = np.array((bottom_line[-1]))
    bottom_right = np.array((bottom_line[0]))

    top_poly = poly(top_points)
    bottom_poly = poly(bottom_points)
    
    top_der = np.polyder(top_poly, 1)
    bottom_der = np.polyder(bottom_poly, 1) #getting edge polynoms
    
    top_l = lambda y: integrate.quad(lambda x: np.sqrt(1 + top_der(x) ** 2),  #length calculating functions
                                     top_left[0], y)[0]
    bottom_l = lambda y: integrate.quad(lambda x: np.sqrt(1 + bottom_der(x) ** 2),
                                        bottom_left[0], y)[0] 

    top_len = top_l(top_right[0])
    bottom_len = bottom_l(bottom_right[0]) 

    left_len = np.sqrt(np.sum(top_left - bottom_left) ** 2)
    right_len = np.sqrt(np.sum(top_right - bottom_right) ** 2)

    width = min(top_len, bottom_len)
    height = min(left_len, right_len)
    
    ratio_x = lambda z : (top_l(z) / top_len)
    ratio_y = lambda z : (bottom_l(z) / bottom_len)

    bot_cur = int(bottom_left[0])

    res = []

    for top in range(int(top_left[0]), int(top_right[0]) + 1): #here biection between top and bottom points is calculated 
        ratio = ratio_x(top)

        while(ratio_y(bot_cur) < ratio):
            bot_cur += 1

        res.append((ratio, np.poly1d(np.polyfit([top, bot_cur], [top_poly(top), bottom_poly(bot_cur)], 1)), top, bot_cur))
        bot_cur += 1
        
        
    h = int(top_left[0] + width + 1) - int(top_left[0])
    w = int(top_left[1] + height + 1) - int(top_left[1])

    final_image = Image.new('RGB', (h, w))
    final_pixels = final_image.load()
    
    pixels = img.load()

    for x in tqdm(range(h)):
        for y in range(w): 
            for ratio, top, bot in res:
                if ( abs(x) / width - ratio < 1e-3):
                    part = y / height
                    lambda_ = part / (1 - part)
                    final_pixels[x, y] = pixels[(top + lambda_ * bot) // (1 + lambda_), 
                                                (top_poly(top) + lambda_ * bottom_poly(bot)) // (1 + lambda_)]
                    break    
    
    
    return final_image

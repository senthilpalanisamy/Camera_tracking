import cv2

def core_logic_point_generateion(add_or_subtract, start_point, start_indices,
                                y_interval=60, x_interval=70, max_pixel_value=1024):


    start_point_x, start_point_y = start_point
    min_y_index = -10
    max_y_index = 10

    current_x_index, current_y_index = start_indices
    start_x_index, start_y_index = start_indices


    #current_y_index = add_or_subtract(current_y_index, -1)
    #current_point_y = add_or_subtract(start_point_y, y_interval)
    current_y_index = start_y_index
    current_point_y = start_point[1]
    all_points = []

    while((min_y_index <= current_y_index <= max_y_index) and 
           0 <= current_point_y < max_pixel_value):

        min_x_index = -20 + abs(current_y_index)
        max_x_index = 20 - abs(current_y_index)

        for current_x_index in range(min_x_index, max_x_index+1):
            current_point_x = start_point_x + (start_x_index - current_x_index) * x_interval
            if((current_x_index % 2 == current_y_index % 2) and 
               (0 < current_point_x < max_pixel_value)):
                all_points.append([current_point_x, current_point_y,
                                   current_x_index, current_y_index])
        #current_y_index = current_y_index - 1
        current_y_index = add_or_subtract(current_y_index, -1)
        current_point_y = add_or_subtract(current_point_y, y_interval)
        #current_point_y = current_point_y - y_interval 
    return all_points

def generate_points(start_indices, start_point, x_interval =34 ,
                    y_interval =63, max_x=1024, max_y=1024):
    all_points = []
    add = lambda x,y:x+y
    subtract = lambda x,y: x-y
    all_points += core_logic_point_generateion(add, start_point, start_indices, y_interval, 
                                 x_interval)
    all_points += core_logic_point_generateion(subtract, start_point, start_indices, y_interval, 
                                  x_interval)
    return all_points


if __name__=='__main__':
    start_indices = [-17,-1]
    start_point = [196, 131]
    all_points = generate_points(start_indices, start_point)
    image_name = 'camera1'
    image = cv2.imread(image_name+'.bmp')
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, cell_x, cell_y in all_points:
        cv2.circle(image, (x,y), 10, (0, 255, 0), -1)
        cv2.putText(image, str(cell_x)+','+str(cell_y), 
                   (x,y-10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(image_name+'result'+'.bmp', image)
    print(all_points)









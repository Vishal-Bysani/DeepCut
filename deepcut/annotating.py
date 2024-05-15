import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    from pygame.locals import *
import sys
import cv2 as cv
import numpy as np


def main(image_path):
    print(f'Annotating {image_path}... ', end='')

    image = pygame.image.load(image_path)
    width, height = image.get_size()

    pygame.init()

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    rectangles = []
    current_rectangle = None
    colors = [(0, 255, 0), (255, 0, 0)]

    while True:
        completed = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if len(rectangles) < 2:
                    current_rectangle = [event.pos, event.pos]
            elif event.type == MOUSEMOTION:
                if current_rectangle:
                    current_rectangle[1] = event.pos
            elif event.type == MOUSEBUTTONUP:
                if current_rectangle:
                    rectangles.append(tuple(current_rectangle))
                    current_rectangle = None
                    if len(rectangles) == 2:
                        completed = True

        screen.blit(image, (0, 0))
        for i, rectangle in enumerate(rectangles):
            rect = pygame.Rect(rectangle[0][0],
                               rectangle[0][1],
                               rectangle[1][0] - rectangle[0][0],
                               rectangle[1][1] - rectangle[0][1])
            pygame.draw.rect(screen, colors[i], rect, 2)
        if current_rectangle:
            rect = pygame.Rect(current_rectangle[0][0],
                               current_rectangle[0][1],
                               current_rectangle[1][0] - current_rectangle[0][0],
                               current_rectangle[1][1] - current_rectangle[0][1])
            pygame.draw.rect(screen, colors[len(rectangles)], rect, 2)

        if completed:
            x_bb_0 = rectangles[0][0][0]
            x_bb_1 = rectangles[0][1][0]
            y_bb_0 = rectangles[0][0][1]
            y_bb_1 = rectangles[0][1][1]

            x_halo_0 = rectangles[1][0][0]
            x_halo_1 = rectangles[1][1][0]
            y_halo_0 = rectangles[1][0][1]
            y_halo_1 = rectangles[1][1][1]

            a = x_bb_0 - x_halo_0
            b = y_bb_0 - y_halo_0
            c = x_halo_1 - x_bb_1
            d = y_halo_1 - y_bb_1

            min_separation = min(a, b, c, d)
            r_a = a / min_separation
            r_b = b / min_separation
            r_c = c / min_separation
            r_d = d / min_separation
            area_bb = (y_bb_1 - y_bb_0) * (x_bb_1 - x_bb_0)

            A = (r_b + r_d) * (r_a + r_c)
            B = ((x_bb_1 - x_bb_0) * (r_b + r_d) + (y_bb_1 - y_bb_0) * (r_a + r_c))
            C = (x_bb_1 - x_bb_0) * (y_bb_1 - y_bb_0) - 2 * area_bb

            min_separation = -B / (2 * A) + np.sqrt(B ** 2 - 4 * A * C) / (2 * A)

            a = r_a * min_separation
            b = r_b * min_separation
            c = r_c * min_separation
            d = r_d * min_separation

            rectangles = np.array([list(item) for rectangle in rectangles for item in rectangle])
            rectangles = np.reshape(rectangles, (2, 2, 2))
            x_halo_0 = int(x_bb_0 - a)
            y_halo_0 = int(y_bb_0 - b)
            x_halo_1 = int(x_bb_1 + c)
            y_halo_1 = int(y_bb_1 + d)
            image_dir = image_path[:image_path.rfind('/images/')]
            image_name = image_path[image_path.rfind('/') + 1:image_path.rfind('.')]
            image = cv.imread(f'{image_dir}/images/{image_name}.png')

            # crop image
            img = image[y_halo_0:y_halo_1, x_halo_0:x_halo_1]
            rectangles[0, 0, 0] -= rectangles[1, 0, 0]
            rectangles[0, 1, 0] -= rectangles[1, 0, 0]
            rectangles[0, 0, 1] -= rectangles[1, 0, 1]
            rectangles[0, 1, 1] -= rectangles[1, 0, 1]
            x_bb_0 -= x_halo_0
            x_bb_1 -= x_halo_0
            y_bb_0 -= y_halo_0
            y_bb_1 -= y_halo_0
            # coords = [str(coord) for rect in rectangles for coord in rect]
            annotation = str(x_bb_0) + ',' + str(y_bb_0) + ',' + str(x_bb_1) + ',' + str(y_bb_1)
            with open(f'{image_dir}/annotations/{image_name}.csv', 'w') as file:
                file.write(annotation)
            print(annotation)
            cv.imwrite(f'{image_dir}/images/{image_name}.png', img)
            print(' Done')
            exit(0)

        pygame.display.flip()
        clock.tick(165)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 annotating.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])

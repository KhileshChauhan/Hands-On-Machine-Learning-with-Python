import python_pb2
snake = python_pb2.Snake()
snake.length_in_inches = 1
snake.circumference_in_inches = 1
snake.color.red = 1
snake.color.green = 1
snake.color.blue = 1
snake.has_rattle = True
snake.venom_index = 1
snake.notes = "notes"
snake.scales_per_inch = 1
snake.species = python_pb2.Snake.PYTHON
snake.territories.append(python_pb2.Snake.RAIN_FOREST)
snake.territories.append(python_pb2.Snake.GRASSLAND)
print(snake)

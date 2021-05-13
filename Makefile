
all:
	g++ -Wall -pedantic -std=c++11 -Wextra LTexture.cpp main.cpp -o app -I /usr/local/include/ -L /usr/local/lib -framework OpenCL -framework GLUT -framework OpenGL -framework Cocoa -framework IOKit -lglfw -lIL -lILU

clean:
	rm -rf *.o
	rm *.x
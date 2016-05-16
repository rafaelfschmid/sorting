// basic file operations
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>
using namespace std;

int main(int argc, char **argv) {

	ifstream input(argv[1]);
	ofstream output(argv[2]);

	string line;
	if (input.is_open()) {
		std::vector<std::vector<double> > multiple_times;

		while (getline(input, line)) {
			std::vector<double> times;

			//cout << "Size: " << line << "\n";
			int size = stoi(line);
			times.push_back(size);

			for (int i = 0; i < 10; i++) {
				getline(input, line);
				//cout << line << "\n";
				times.push_back(stod(line));
			}
			multiple_times.push_back(times);

			getline(input, line);
		}
		input.close();

		for (int j = 0; j < 11; j++) {
			for (int i = 0; i < multiple_times.size(); i++) {
				output << std::fixed << multiple_times[i][j] << "\t";
				//cout << std::fixed << multiple_times[i][j] << ";";
			}
			output << "\n";
			//cout << "\n";
		}
		output.close();
	}

	return 0;
}

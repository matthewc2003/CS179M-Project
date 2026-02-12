#include <iostream>
#include <fstream>
#include <random>

/*The purpose of this program is to randomly generate 1000 records of people with features of age, weight,
diabetes, exercise, and diet. This data is not accurate, nor is it supposed to be; however, it does somewhat
aim to be vaguely feature-dependent.*/

int main() {
    srand(0);
    std::ofstream outData("dummydata.csv");
    if (!outData.is_open()) exit(EXIT_FAILURE);

    int age, weight, diabetes, exercises, eats_well, rand_score;
    outData << "age,weight,diabetes,exercises,eats well\n";
    for (unsigned i = 0; i < 1000; i++) {
        age = rand() % 100;

        if (age < 13) weight = rand() % 100;
        else weight = rand() % 300 + 100;

        rand_score = rand() % 100;
        if (weight < 200) {
            if (rand_score > 40) exercises = 1;
            else exercises = 0;
        }
        else {
            if (rand_score > 80) exercises = 1;
            else exercises = 0;
        }

        rand_score = rand() % 100;
        if (exercises) {
            if (rand_score > 10) eats_well = 1;
            else eats_well = 0;
        }
        else if (weight < 200) {
            if (rand_score > 60) eats_well = 1;
            else eats_well = 0;
        }
        else {
            if (rand_score > 90) eats_well = 1;
            else eats_well = 0;
        }

        rand_score = rand() % 100;
        if (rand_score - weight / 4 < 0) {
            diabetes = 1;
            rand_score = rand() % 100;
            if (eats_well && rand_score > 40) diabetes = 0;
            rand_score = rand() % 100;
            if (exercises && rand_score > 40) diabetes = 0;
        }
        else diabetes = 0;

        outData << age << ',' << weight << ',' << diabetes << ',' << exercises << ',' << eats_well << std::endl;
    }
    outData.close();
}
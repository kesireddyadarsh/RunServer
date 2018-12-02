//
//  main.cpp
//  rover_domain
//
//  Created by ak on 10/11/18.
//  Copyright © 2018 ak. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <cmath>
#include <math.h>
#include <random>
#include <assert.h>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define PI 3.14159265



/*************************
 Neural Network
 ************************/

struct connect{
    double weight;
};

/********************************************
 Fucntion generates random number
 *********************************************/

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron
class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex,int numNN);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
    
};

/********************************************
 //This creates connection with neurons.weight is weight of each connection
 *********************************************/
Neuron::Neuron(unsigned numOutputs, unsigned myIndex,int number){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        double Min = 0.5;
        double Max = 1.0;
        switch (number) {
            case 0:
                Min = 0.75;
                Max = 1.0;
                break;
            case 1:
                Min = 0.50;
                Max = 0.75;
                break;
            case 2:
                Min = 0.25;
                Max = 0.50;
                break;
            case 3:
                Min = 0.0;
                Max = 0.25;
                break;
            case 4:
                Min = -0.25;
                Max = 0.0;
                break;
            case 5:
                Min = -0.50;
                Max = -0.25;
                break;
            case 6:
                Min = -0.75;
                Max = -0.50;
                break;
            case 7:
                Min = -1.0;
                Max = -0.75;
                break;
            case 8:
                Min = -1.0;
                Max = 1.0;
                break;
            default:
                break;
        }
        
        double random_number = (((double) rand() / RAND_MAX) * (Max-Min)) +Min;
        FILE* p_weights_neuron;
        p_weights_neuron = fopen("neuron", "a");
        fprintf(p_weights_neuron, "%d \t %f \t",number, random_number);
        fclose(p_weights_neuron);
        
        z_outputWeights.back().weight = (random_number);
        // ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
    }
    z_myIndex = myIndex;
}

/********************************************
 //Function for activation function. We currently have 3 but using tanh function
 *********************************************/
double Neuron::transferFunction(double x){
    int case_to_use = 1;
    switch (case_to_use) {
        case 1:
            if (x<0) {
                x= -sin(sqrt(-x));
//                x= -cos(sqrt(-x));
//                x= -sin(-x);
//                x= -cos(-x);
            }else{
                x= sin(sqrt(x));
//                x= cos(sqrt(x));
//                x= sin(x);
//                x= cos(x);
            }
            return x;
            break;
        case 2:
            //Dont use this case
            return 1/(1+exp(x));
            break;
        case 3:
            return x/(1+abs(x));
            break;

        default:
            break;
    }

    return tanh(x);

}

/********************************************
 // In this function weight*value is performed.
 //In addition weight*value is summed up
 //After that passed through activation function
 *********************************************/

void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
//        cout << prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight<<endl;
//        cout << prevLayer[n].z_outputWeights[z_myIndex].weight<<endl;
//        cout << prevLayer[n].getOutputVal()<<endl;
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
//        cout<<"This is sum value"<<sum<<endl;
    }

    
    //Normalize sum value between -1 to 1 https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
//    double min_x = -1;
//    double max_x = 1;
//    sum = ((sum-min_x)/(max_x-min_x));
////    sum = (2*((sum-min_x)/(max_x-min_x)))-1;
    
   // cout<<sum<<endl;
    
    z_outputVal = Neuron::transferFunction(sum);
    
//    cout<<z_outputVal<<endl;
}

/********************************************
 //This is single neural network
 *********************************************/
class Net{
public:
    Net(vector<unsigned> topology,int numNN,int number);
    void feedForward(vector<double> inputVals);
    vector<Layer> z_layer;
    vector<double> outputvaluesNN;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<double> z_error_vector;
    void mutate();
    vector<double> temp_inputs;
    vector<double> temp_targets;
    
    //coordinates
    vector<double> x_coordinates;
    vector<double> y_coordinates;
    
    vector<double> target_distance;//Distance to target
    double shortest_target_distance; // Shortest distance to target
    vector<vector<double>> obstacle_distance; //Saves all obstacle distances
    double hitting_obstacle;// Hitting of obstacle all of them
    vector<vector<double>> right_rover; // Right rover distance values
    vector<vector<double>> left_rover; //left rover distance values
    
    vector<double> hitting_right_rover; // Hitting right rover
    vector<double> hitting_left_rover; // Hitting left rover
};

/********************************************
 //Here we are creating neural network with given topology
 //Topology is given in main function
 *********************************************/
Net::Net(vector<unsigned> topology,int numNN,int number){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>30){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        
        z_layer.push_back(Layer());
        
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons,number));
        }
        z_layer.back().back().setOutputVal(1.0);
    }
}


/********************************************
 //This function is used in evolutionary algorithm
 //We are changing weights of all connections in the following
 *********************************************/
void Net::mutate(){
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer.at(l).size(); n++) {
            for (int z=0 ; z< z_layer.at(l).at(n).z_outputWeights.size(); z++) {
                //                z_layer.at(l).at(n).z_outputWeights.at(z).weight = (random_global(.5)-random_global(.5));
                //                z_layer.at(l).at(n).z_outputWeights.at(z).weight += ((((double) rand() / RAND_MAX) * 2) - 1.0);
                z_layer.at(l).at(n).z_outputWeights.at(z).weight += (random_global(.05)-random_global(.05));
            }
        }
    }
}

/********************************************
 //Neuron feedforward does only for one neuron. Here we are doing for entire neural network
 //Each neuron feedforward is called from this function
 *********************************************/
void Net::feedForward(vector<double> inputVals){
    
    assert(inputVals.size() == z_layer[0].size()-1);
    for (unsigned i=0; i<inputVals.size(); ++i) {
        z_layer[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
        Layer &prevLayer = z_layer[layerNum - 1];
        for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
            z_layer[layerNum][n].feedForward(prevLayer);
        }
    }
    temp_inputs.clear();
    Layer &outputLayer = z_layer.back();
    z_error_temp = 0.0;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        //cout<<"This is value from outputlayer.getourputvalue:::::"<<outputLayer[n].getOutputVal()<<endl;
        //double delta = temp_targets[n] - outputLayer[n].getOutputVal();
        //cout<<"This is delta value::"<<delta;
        //z_error_temp += delta * delta;
        outputvaluesNN.push_back(outputLayer[n].getOutputVal());
    }
    
    //Normalize output values
//    for (int i = 0; i< outputvaluesNN.size(); i++) {
////        double min_x = -5000;
////        double max_x = 5000;
////        outputvaluesNN.at(i) = (2*((outputvaluesNN.at(i)-min_x)/(max_x-min_x)))-1;
////        double min_x = -1;
////        double max_x = 1;
////        outputvaluesNN.at(i) = (2*((outputvaluesNN.at(i)-min_x)/(max_x-min_x)))-1;
//        outputvaluesNN.at(i) = (outputvaluesNN.at(i)*2)-1; // a= (a*(max-min))+min
//    }
    
//    for (int i = 0; i< outputvaluesNN.size(); i++) {
//        cout<<outputvaluesNN.at(i)<<"\t";
//    }
//    cout<<endl;
    
}

/********************************************
 We are not using this function. This is just for my testing to print out error
 *********************************************/
double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<"This is z_error_vector"<<temp<<" value::"<< z_error_vector[temp]<<endl;
        z_error += z_error_vector[temp];
    }
    //    cout<<"This is z_error::"<<z_error<<endl;
    return z_error;
}


/**************************
 New Rover
 **************************/

class new_rover{
public:
    double x_location_new,y_location_new;
    double x_const,y_const;
    double target_x,target_y;
    double theta_new,phi_new;
    int team_number;
    vector<double> x_coordinates;
    vector<double> y_coordinates;
    vector<double> sensor;
    void create_nn(int numNN, vector<unsigned> topology);
    vector<Net> new_network;
    void sense_new_rover(double x,double y);
    void sense_new_target(double x, double y);
    void sense_new_ob(double x, double y);
    int quad_value(double difference_x, double difference_y);
    void reset_sensor_value();
    int front_number;
    int ranking_number;
    double crowding_distance;
    double consenses; // collision with agents and formation
    double collision; // collision with obstacle
    double target; // reaching destination
    double total_reward; //summation of all rewards
    vector<double> right_rover;
    vector<double> left_rover;
    vector<double> target_distance;
    vector<vector<double>> obstacle_distance;
    void set_sensor_zero();
    
    vector<double> difference_reward;
    
};


// variables used: indiNet -- object to Net
void new_rover::create_nn(int numNN,vector<unsigned> topology){
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology,numNN,populationNum);
        new_network.push_back(singleNetwork);
    }
}

void new_rover::reset_sensor_value(){
    sensor.clear();
    if (sensor.size()!= 12) {
        for (int i=0; i<12; i++) {
            sensor.push_back(0.0);
        }
    }
}

void new_rover::set_sensor_zero(){
    for (int i=0; i<12; i++) {
        sensor.push_back(0.0);
    }
}


int new_rover::quad_value(double difference_x, double difference_y){
    int value = 0;
    
    if ((difference_x == 0)&&(difference_y == 0)) {
        value = 0;
    }else if (difference_x == 0){
        if (difference_y > 0) {
            value = 5;
        }else{
            value = 7;
        }
    }else if (difference_y ==0){
        if (difference_x > 0) {
            value = 8;
        }else{
            value = 6;
        }
    }else{
        if (difference_x>0) {
            if (difference_y>0) {
                value = 1;
            }else{
                value =4;
            }
        }else{
            if (difference_y>0) {
                value =2;
            }else{
                value =3;
            }
        }
    }
    return value;
}

void new_rover::sense_new_rover(double x, double y){
    double difference_x = x-x_location_new;
    double difference_y = y-y_location_new;
    double distance_rover = sqrt(pow(difference_x, 2)+pow(difference_y, 2));
    //    cout<<"distance_rover"<<endl;
    //    for (int sense = 0 ; sense < sensor.size(); sense++) {
    //        cout<<sensor.at(sense)<<"\t";
    //    }
    //    cout<<" Here \n";
    int quad = quad_value(difference_x, difference_y);
    if ((quad == 0)||(quad == 8) || (quad == 1)) {
        sensor.at(0) += distance_rover;
    }else if ((quad ==5)||(quad ==2)){
        sensor.at(3) += distance_rover;
    }else if ((quad == 6)||(quad == 3)){
        sensor.at(6) += distance_rover;
    }else{
        sensor.at(9) +=distance_rover;
    }
}

void new_rover::sense_new_target(double x, double y){
    double difference_x = x-x_location_new;
    double difference_y = y-y_location_new;
    double distance_rover = sqrt(pow(difference_x, 2)+pow(difference_y, 2));
    int quad = quad_value(difference_x, difference_y);
    if ((quad == 0)||(quad == 8) || (quad == 1)) {
        sensor.at(1) += distance_rover;
    }else if ((quad ==5)||(quad ==2)){
        sensor.at(4) += distance_rover;
    }else if ((quad == 6)||(quad == 3)){
        sensor.at(7) += distance_rover;
    }else{
        sensor.at(10) +=distance_rover;
    }
}

void new_rover::sense_new_ob(double x, double y){
    double difference_x = x-x_location_new;
    double difference_y = y-y_location_new;
    double distance_rover = sqrt(pow(difference_x, 2)+pow(difference_y, 2));
    int quad = quad_value(difference_x, difference_y);
    if ((quad == 0)||(quad == 8) || (quad == 1)) {
        sensor.at(2) += distance_rover;
    }else if ((quad ==5)||(quad ==2)){
        sensor.at(5) += distance_rover;
    }else if ((quad == 6)||(quad == 3)){
        sensor.at(8) += distance_rover;
    }else{
        sensor.at(11) +=distance_rover;
    }
}


/********************************************************
 Population of each team
 *******************************************************/
class population{
public:
    population(int number_of_rover, int number_of_routes);
    vector<new_rover> teamRover;
    vector<double> objective_values;
};



population::population(int number_of_rover, int number_of_routes){
    //This is for neural network
    vector<unsigned> topology;
    topology.clear();
    topology.push_back(12);
    topology.push_back(7);
    topology.push_back(2);
    new_rover a;
    
    //This created neural network on the rover
    for (int number = 0 ; number < number_of_rover; number++) {
        teamRover.push_back(a);
        teamRover.at(number).create_nn(number_of_routes, topology);
    }
    
}

/********************************************************
 This function is to calculate scalarization of neural network
 ********************************************************/

void scalarization(vector<new_rover> teamRover,int rover){
    double min = 9999.99;
    double max = -9999999.99;
    for (int sense = 0; sense < teamRover.at(rover).sensor.size(); sense++) {
        if (teamRover.at(rover).sensor.at(sense) < min) {
            min = teamRover.at(rover).sensor.at(sense);
        }
        
        if (teamRover.at(rover).sensor.at(sense) > max) {
            max = teamRover.at(rover).sensor.at(sense);
        }
    }
    
    for (int sense = 0; sense < teamRover.at(rover).sensor.size(); sense++) {
        double temp = teamRover.at(rover).sensor.at(sense);
        teamRover.at(rover).sensor.at(sense) = ((temp -min)/(max -min));
    }
    
}


/********************************************************
 This function  is to calculate distance between two points
 ********************************************************/

double cal_distance(double x1, double y1, double x2, double y2){
    return sqrt(pow((x1 - x2), 2)+pow((y1 - y2), 2));
}


/********************************************************
 This function initialize all the environment
 ********************************************************/


void initial_team(vector<population>* teams,vector<vector<double>>* location_obstacle,int number_of_obstacles, vector<vector<double>>* p_stat, double distance_between_rover){
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0; rover < teams->at(team_number).teamRover.size(); rover++) {
            //Setting target location here
            teams->at(team_number).teamRover.at(rover).target_x = 2.0+(distance_between_rover*rover);
            teams->at(team_number).teamRover.at(rover).target_y = 50.0;
            teams->at(team_number).teamRover.at(rover).x_coordinates.clear();
            teams->at(team_number).teamRover.at(rover).y_coordinates.clear();
            teams->at(team_number).teamRover.at(rover).target_distance.clear();
            teams->at(team_number).teamRover.at(rover).obstacle_distance.clear();
            teams->at(team_number).teamRover.at(rover).right_rover.clear();
            teams->at(team_number).teamRover.at(rover).left_rover.clear();
            teams->at(team_number).teamRover.at(rover).target = 0;
            teams->at(team_number).teamRover.at(rover).consenses = 0;
            teams->at(team_number).teamRover.at(rover).collision = 0;
            teams->at(team_number).teamRover.at(rover).team_number = team_number;
            
        }
        
        
        
        //reseting the rover to initial location
        for (int rover = 0 ; rover <teams->at(team_number).teamRover.size(); rover++) {
            teams->at(team_number).teamRover.at(rover).x_location_new = p_stat->at(rover).at(0);
            teams->at(team_number).teamRover.at(rover).y_location_new = p_stat->at(rover).at(1);
        }
        
        for (int ob = 0 ; ob < number_of_obstacles; ob++) {
            double rand_1 = random_global(750);
            double rand_2 = random_global(750);
            
            if(ob == 0){
                rand_1 = 10;
                rand_2 = 8;
            }else if (ob == 1){
                rand_1 = 15;
                rand_2 = 20;
            }else if (ob == 2){
                rand_1 = 20;
                rand_2 = 15;
            }
            vector<double> temp;
            temp.push_back(rand_1);
            temp.push_back(rand_2);
            
            //Check if rand_1 or rand_2 is on or near
            location_obstacle->push_back(temp);
            temp.clear();
        }
    }
}



/*******************************************************
This function runs through each simulation
 *******************************************************/

void simulation_team(vector<population>* teams, vector<vector<double>>* p_location_obstacle,int generation,int number_of_obstacles, vector<vector<double>>* p_stat, double distance_between_rover){
    int max_time_step = 55;
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int neuralNetwork = 0 ; neuralNetwork< teams->at(team_number).teamRover.at(0).new_network.size(); neuralNetwork++) {
            initial_team(teams, p_location_obstacle,number_of_obstacles,p_stat, distance_between_rover);
            for (int time_step = 0 ; time_step < max_time_step ; time_step++) {
                for (int rover = 0; rover < teams->at(team_number).teamRover.size(); rover++) {
                    
                    //Save the x and y coordinates
                    teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).x_coordinates.push_back(teams->at(team_number).teamRover.at(rover).x_location_new);
                    teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).y_coordinates.push_back(teams->at(team_number).teamRover.at(rover).y_location_new);
                    
                    //Reset sensor values
                    teams->at(team_number).teamRover.at(rover).reset_sensor_value();
                    
                    //sense all other rovers
                    for (int other_rover = 0 ; other_rover < teams->at(team_number).teamRover.size(); other_rover++) {
                        if (rover != other_rover) {
                            teams->at(team_number).teamRover.at(rover).sense_new_rover(teams->at(team_number).teamRover.at(other_rover).x_location_new, teams->at(team_number).teamRover.at(other_rover).y_location_new);
                        }
                    }
                    
                    //sense target location
                    teams->at(team_number).teamRover.at(rover).sense_new_target(teams->at(team_number).teamRover.at(rover).target_x, teams->at(team_number).teamRover.at(rover).target_y);
                    
                    //sense obstacles
                    for (int obstacle = 0; obstacle< p_location_obstacle->size(); obstacle++) {
                        teams->at(team_number).teamRover.at(rover).sense_new_ob(p_location_obstacle->at(obstacle).at(0), p_location_obstacle->at(obstacle).at(1));
                    }
                    
                    teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).outputvaluesNN.clear();

                    FILE* p_sensor;
                    p_sensor = fopen("Sensor_values", "a");
                    for (int i = 0 ; i < teams->at(team_number).teamRover.at(rover).sensor.size(); i++) {
                        fprintf(p_sensor, "%f \t",teams->at(team_number).teamRover.at(rover).sensor.at(i));
                    }
                    fprintf(p_sensor, "\n");
                    fclose(p_sensor);
                    
                    
                    //Pass sensor values to neural network and obtain new teleportation point
                    teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).feedForward(teams->at(team_number).teamRover.at(rover).sensor);
                    scalarization(teams->at(team_number).teamRover, rover);
                    
                    
                    //Out put from neural network
                    double dx = teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).outputvaluesNN.at(0);
                    double dy = teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).outputvaluesNN.at(1);
                    teams->at(team_number).teamRover.at(rover).new_network.at(neuralNetwork).outputvaluesNN.clear();
                    teams->at(team_number).teamRover.at(rover).sensor.clear();
                    
                    //New location of rovers
                    teams->at(team_number).teamRover.at(rover).x_location_new += dx;
                    teams->at(team_number).teamRover.at(rover).y_location_new += dy;
                    
                }
            }
        }
    }
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size() == max_time_step);
                assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size() == teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.size() );
            }
        }
    }
}


/********************************************************
 This function is to check if both x_1 and x_2 are in same coordinates.
 ********************************************************/

bool check_quad(double x_1, double x_2){
    if ((x_1 >=0)&& (x_2>=0)) {
        return true;
    }else if ((x_1 < 0)&& (x_2 <0)){
        return  true;
    }
    
    return false;
    
}


/********************************************************
 This function is to calculate the following:
 1. Distance between each rover
 2. Distance to the target
 3. Distance to each obstacle
 ********************************************************/


void distance_team(vector<population>* teams, double distance_between_rover, double safe_distance_between_rover, double radius_of_obstacle, vector<vector<double>>* p_location_obstacle){
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover =  0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            
            for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                //distance to target
                for (int index = 0; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); index++) {
                    teams->at(team_number).teamRover.at(rover).new_network.at(neural).target_distance.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).target_x, teams->at(team_number).teamRover.at(rover).target_y, teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index)));
                }
                
                //each obstacle distance
                for (int index = 0 ; index < p_location_obstacle->size(); index++) {
                    vector<double> temp_distance;
                    for (int x = 0; x < teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); x++) {
                        temp_distance.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(x), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(x), p_location_obstacle->at(index).at(0), p_location_obstacle->at(index).at(1)));
                    }
                    teams->at(team_number).teamRover.at(rover).new_network.at(neural).obstacle_distance.push_back(temp_distance);
                    temp_distance.clear();
                }
            }
            if (rover == 0) {
                int right_rover = rover + 1;
                for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                    vector<double> temp_distance;
                    for (int right_neural = 0 ; right_neural < teams->at(team_number).teamRover.at(right_rover).new_network.size(); right_neural++) {
                        for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); index++) {
                            temp_distance.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index), teams->at(team_number).teamRover.at(right_rover).new_network.at(right_neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(right_rover).new_network.at(right_neural).x_coordinates.at(index)));
                        }
                        teams->at(team_number).teamRover.at(rover).new_network.at(neural).right_rover.push_back(temp_distance);
                        temp_distance.clear();
                    }
                }
            }else if (rover == (teams->at(team_number).teamRover.size()-1)) {
                int left_rover = rover - 1;
                for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                    vector<double> temp_distance;
                    for (int left_neural = 0 ; left_neural < teams->at(team_number).teamRover.at(left_rover).new_network.size(); left_neural++) {
                        for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); index++) {
                            temp_distance.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index), teams->at(team_number).teamRover.at(left_rover).new_network.at(left_neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(left_rover).new_network.at(left_neural).x_coordinates.at(index)));
                        }
                        teams->at(team_number).teamRover.at(rover).new_network.at(neural).left_rover.push_back(temp_distance);
                        temp_distance.clear();
                    }
                }
            }else{
                int right_rover = rover + 1;
                int left_rover = rover - 1;
                for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                    for (int neural_next = 0 ; neural_next < teams->at(team_number).teamRover.at(right_rover).new_network.size(); neural_next++) {
                        vector<double> temp_left;
                        vector<double> temp_right;
                        for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); index++) {
                            temp_left.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index), teams->at(team_number).teamRover.at(left_rover).new_network.at(neural_next).x_coordinates.at(index), teams->at(team_number).teamRover.at(left_rover).new_network.at(neural_next).y_coordinates.at(index)));
                            temp_right.push_back(cal_distance(teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index), teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index), teams->at(team_number).teamRover.at(right_rover).new_network.at(neural_next).x_coordinates.at(index), teams->at(team_number).teamRover.at(right_rover).new_network.at(neural_next).y_coordinates.at(index)));
                        }
                        teams->at(team_number).teamRover.at(rover).new_network.at(neural).left_rover.push_back(temp_left);
                        teams->at(team_number).teamRover.at(rover).new_network.at(neural).right_rover.push_back(temp_right);
                        temp_right.clear();
                        temp_left.clear();
                    }
                }
            }
        }
    }
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                if (rover == 0) {
                    assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).left_rover.size() == 0);
                }else if (rover == (teams->at(team_number).teamRover.size() -1)){
                    assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).right_rover.size() == 0);
                }else{
                    assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).left_rover.size() == teams->at(team_number).teamRover.at(rover).new_network.at(neural).right_rover.size());
                }
                
                assert(teams->at(team_number).teamRover.at(rover).new_network.at(neural).obstacle_distance.size() == p_location_obstacle->size());
            }
        }
    }
    
    
}


/********************************************************
 This function is to calculate the reward structure of all agents
 ********************************************************/



void reward_team(vector<population>* teams, double distance_between_rover, double safe_distance_between_rover, double radius_of_obstacle, int number_of_objectives){
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            for (int neural_network = 0 ; neural_network < teams->at(team_number).teamRover.at(rover).new_network.size(); neural_network++) {
                
                teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).hitting_obstacle = 0.0;
                
                teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).shortest_target_distance = *std::min_element(teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).target_distance.begin(), teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).target_distance.end());
                
                if (teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).shortest_target_distance > 2) {
                    teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).shortest_target_distance += 1000;
                }
                
                //Hitting a obstacle
                for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).obstacle_distance.size(); index++) {
                    for (int index_1 = 0 ; index_1 < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).obstacle_distance.at(index).size(); index_1++) {
                        if (teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).obstacle_distance.at(index).at(index_1) < radius_of_obstacle) {
                            teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).hitting_obstacle += 1000;
                        }
                    }
                }
                
                //Hitting other agent
                for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).right_rover.size(); index++) {
                    double temp_hitting = 0.0;
                    for (int index_1 = 0 ; index_1 < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).right_rover.at(index).size(); index_1++) {
                        if (teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).right_rover.at(index).at(index_1) < safe_distance_between_rover) {
                            temp_hitting += 1000;
                        }
                    }
                    teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).hitting_right_rover.push_back(temp_hitting);
                }
                for (int index = 0 ; index < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).left_rover.size(); index++) {
                    double temp_hitting = 0.0;
                    for (int index_1 = 0 ; index_1 < teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).left_rover.at(index).size(); index_1++) {
                        if (teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).left_rover.at(index).at(index_1) < safe_distance_between_rover) {
                            temp_hitting += 1000;
                        }
                    }
                    teams->at(team_number).teamRover.at(rover).new_network.at(neural_network).hitting_left_rover.push_back(temp_hitting);
                }
            }
        }
    }
    /*
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        
        //Target location
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            teams->at(team_number).teamRover.at(rover).target = 0.0;
            teams->at(team_number).teamRover.at(rover).collision = 0.0;
            teams->at(team_number).teamRover.at(rover).consenses = 0.0;
            double low = 9999999.999;
            
            for (int x = 0; x< teams->at(team_number).teamRover.at(rover).target_distance.size(); x++) {
                if (low > teams->at(team_number).teamRover.at(rover).target_distance.at(x)) {
                    low = teams->at(team_number).teamRover.at(rover).target_distance.at(x);
                }
            }
            
            for (int x = 0 ; x < teams->at(team_number).teamRover.at(rover).target_distance.size(); x++) {
                teams->at(team_number).teamRover.at(rover).target = low;
            }
            
            for (int x = 0; x < teams->at(team_number).teamRover.at(rover).target_distance.size(); x++) {
                bool check_x = check_quad(teams->at(team_number).teamRover.at(rover).x_coordinates.at(x), teams->at(team_number).teamRover.at(rover).target_x);
                bool check_y = check_quad(teams->at(team_number).teamRover.at(rover).y_coordinates.at(x), teams->at(team_number).teamRover.at(rover).target_y);
                
                if (!check_x) {
                    teams->at(team_number).teamRover.at(rover).target +=10;
                }
                if (!check_y) {
                    teams->at(team_number).teamRover.at(rover).target +=10;
                }
            }
            
        }
        
        //Obstacles
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            teams->at(team_number).teamRover.at(rover).collision = 0;
            for (int x = 0; x< teams->at(team_number).teamRover.at(rover).obstacle_distance.size(); x++) {
                for (int y = 0; y<teams->at(team_number).teamRover.at(rover).obstacle_distance.at(x).size(); y++) {
                    ///teamRover.at(rover).collision += (1/teamRover.at(rover).obstacle_distance.at(x).at(y));
                    if (radius_of_obstacle > teams->at(team_number).teamRover.at(rover).obstacle_distance.at(x).at(y)) {
                        teams->at(team_number).teamRover.at(rover).collision +=100.0;
                    }
                    
                }
            }
        }
        
        //rover distance and collision
        for (int rover = 0; rover< teams->at(team_number).teamRover.size(); rover++) {
            teams->at(team_number).teamRover.at(rover).consenses = 0;
            if (teams->at(team_number).teamRover.at(rover).left_rover.size() != 0) {
                for (int x =0 ; x< teams->at(team_number).teamRover.at(rover).left_rover.size(); x++) {
                    if ((teams->at(team_number).teamRover.at(rover).left_rover.at(x) >= distance_between_rover ) && (teams->at(team_number).teamRover.at(rover).left_rover.at(x) <= safe_distance_between_rover)){
                        //teamRover.at(rover).consenses += 0.1;
                        teams->at(team_number).teamRover.at(rover).consenses += 0.0;
                    }else{
                        //                    teamRover.at(rover).consenses +=teamRover.at(rover).left_rover.at(x); Change it to 100
                        teams->at(team_number).teamRover.at(rover).consenses +=100.0;
                    }
                }
            }
            if (teams->at(team_number).teamRover.at(rover).right_rover.size() != 0) {
                for (int x =0 ; x< teams->at(team_number).teamRover.at(rover).right_rover.size(); x++) {
                    if ((teams->at(team_number).teamRover.at(rover).right_rover.at(x) >= distance_between_rover ) && (teams->at(team_number).teamRover.at(rover).right_rover.at(x) <= safe_distance_between_rover)){
                        //teamRover.at(rover).consenses += 0.1;
                        teams->at(team_number).teamRover.at(rover).consenses += 0.0;
                    }else{
                        teams->at(team_number).teamRover.at(rover).consenses +=teams->at(team_number).teamRover.at(rover).right_rover.at(x);
                    }
                }
            }
        }
        
        
        //Summation
        for (int rover = 0 ;rover < teams->at(team_number).teamRover.size(); rover++) {
            //        teamRover.at(rover).consenses = 0.0;
            //        teamRover.at(rover).collision = 0.0;
            //        cout<<teamRover.at(rover).consenses<<"\t"<<teamRover.at(rover).collision<<"\t"<<teamRover.at(rover).target<<endl;
            teams->at(team_number).teamRover.at(rover).total_reward = teams->at(team_number).teamRover.at(rover).consenses + teams->at(team_number).teamRover.at(rover).collision + teams->at(team_number).teamRover.at(rover).target;
        }
        
        
        //objective
        for (int objective = 0 ; objective < number_of_objectives; objective++) {
            double value =0.0;
            for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
                if (objective == 0) {
                    value += teams->at(team_number).teamRover.at(rover).collision;
                }else if (objective == 1){
                    value += teams->at(team_number).teamRover.at(rover).consenses;
                }else{
                    value += teams->at(team_number).teamRover.at(rover).target;
                }
            }
            teams->at(team_number).objective_values.push_back(value);
        }
        
        assert(teams->at(team_number).objective_values.size() == number_of_objectives);
        
    }
    
    //Calculate difference reward
    for(int team_number = 0; team_number < teams->size(); team_number++){
        for (int objective = 0 ; objective < number_of_objectives; objective++) {
            for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
                double value = 0.0;
                for (int other_rover = 0 ; other_rover < teams->at(team_number).teamRover.size(); other_rover++) {
                    if (rover == other_rover) {
                        value += teams->at(team_number).teamRover.at(other_rover).collision;
                    }
                }
                teams->at(team_number).teamRover.at(rover).difference_reward.push_back(value);
            }
        }
    }
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            assert(teams->at(team_number).teamRover.at(rover).difference_reward.size() == number_of_objectives);
        }
    }
    */
    
}


void ea(vector<population>* teams){
    int number_of_rover = teams->at(0).teamRover.size();
    int number_of_routes = teams->at(0).teamRover.at(0).new_network.size();
    
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0; rover < teams->at(team_number).teamRover.size(); rover++) {
            for (int route = 0 ; route < (number_of_routes/2); route++) {
                int rand_1 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                int rand_2 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                while (rand_1 == rand_2) {
                    rand_1 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                    rand_2 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                }
                
                if ((teams->at(team_number).teamRover.at(rover).new_network.at(rand_1).shortest_target_distance + teams->at(team_number).teamRover.at(rover).new_network.at(rand_1).hitting_obstacle ) < (teams->at(team_number).teamRover.at(rover).new_network.at(rand_2).shortest_target_distance + teams->at(team_number).teamRover.at(rover).new_network.at(rand_2).hitting_obstacle) ){
                    //remove rand_2
                    teams->at(team_number).teamRover.at(rover).new_network.erase(teams->at(team_number).teamRover.at(rover).new_network.begin()+rand_2);
                }else{
                    //remove rand_2
                    teams->at(team_number).teamRover.at(rover).new_network.erase(teams->at(team_number).teamRover.at(rover).new_network.begin()+rand_1);
                }
            }
            assert(teams->at(team_number).teamRover.at(rover).new_network.size() == (number_of_routes/2));
            
            
            for (int route = (number_of_routes/2); route < number_of_routes; route++) {
                int rand_1 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                teams->at(team_number).teamRover.at(rover).new_network.push_back(teams->at(team_number).teamRover.at(rover).new_network.at(rand_1));
            }
            
            for (int route = 0 ; route < (number_of_routes/2); route++) {
                int rand_1 = rand()%teams->at(team_number).teamRover.at(rover).new_network.size();
                teams->at(team_number).teamRover.at(rover).new_network.at(rand_1).mutate();
            }
        }
    }
}


/************************************
 NSGA II
 *************************************/

vector<vector<double>> min_max;
/*
 
 
 int Population::fastNonDominatedSort(int start, int end){
 int numberProcessed = 0;
 int popSize = end - start;
 int rank = 0;
 resetFronts();
 resetDominationCounts(start, end);
 
 for(int i = start; i<end; i++){
 solutionDominatedByIndex(i, start, end);
 }
 
 while(numberProcessed < popSize){
 int n = findFrontWithRank(rank, start, end);
 this->nInParetoFrontWithRank[rank] = n;
 if (n == 0 && rank > popSize) {
 std::cout << "Found: " << n << " individuals with rank: " << rank << "!!!" << std::endl;
 break;
 }
 
 numberProcessed += n;
 for(int i =0; i<n; i++){
 for(int j = 0; j< pop[fronts[rank][i]]->dominatesCount; j++){
 pop[pop[fronts[rank][i]]->dominates[j]]->dominatedByCount--; // How ??
 }
 }
 rank++;
 }
 return rank;
 }
 
 
 bool Population::comparePareto(const Individual *l1, const Individual *l2, Options opt){
 if(opt.maximize){
 return compareParetoMax(l1, l2);
 }
 else{
 return compareParetoMin(l1, l2);
 }
 }
 
 void Population::solutionDominatedByIndex(int index, int start, int end){
 assert(index >= start && index <end && start<end);
 Individual *me = pop[index];
 me->dominatesCount = 0;
 for(int i = start; i<end; i++){
 if(i == index) continue;
 if(comparePareto(me, pop[i], opt)){
 me->dominates[me->dominatesCount] = i;
 me->dominatesCount++;
 pop[i]->dominatedByCount++;
 }
 }
 }
 
 bool Population::compareParetoMax(const Individual *l1, const Individual *l2){
 int nonInferior = 0;
 int dominant = 0;
 for(int i = 0; i<l1->nCriteria; i++){
 if(l1->fitness[i] > l2->fitness[i]){
 dominant++;
 }
 if(l1->fitness[i] >= l2->fitness[i]){
 nonInferior++;
 }
 }
 return (nonInferior >= l1->nCriteria && dominant > 0);
 }
 
 int Population::findFrontWithRank(int rank, int start, int end){
 int numberInRank = 0;
 for(int i = start; i<end; i++){
 if(pop[i]->dominatedByCount == 0){ // dominatedByCount == 0 means rank0 when call this function first time
 fronts[rank][numberInRank++] = i;
 pop[i]->dominatedByCount = -1 -rank; // update this value to -1-rank, to not count again when we call this function
 pop[i]->Rank = rank;
 }
 }
 return numberInRank;
 }
 
 void Population::computeParetoCrowding(int start, int end){
 double minf, maxf, maxDist;
 int nCriteria = pop[start]->nCriteria;
 for(int i = 0; i<nCriteria; i++){
 sortByCriteria(i, start, end);
 minf = pop[crowds[i][start]]->fitness[i];
 maxf = pop[crowds[i][end - 1]]->fitness[i];
 maxDist = sqrt(pow((maxf-minf), 2.0));
 
 pop[crowds[i][start]]->paretoCrowdingDistance[i] = 1.0;
 pop[crowds[i][end - 1]]->paretoCrowdingDistance[i] = 1.0;
 
 for(int j = start + 1; j<end -1; j++){
 pop[crowds[i][j]]->paretoCrowdingDistance[i] = sqrt(pow((pop[crowds[i][j-1]]->fitness[i] - pop[crowds[i][j+1]]->fitness[i]), 2.0))/maxDist;
 }
 }
 for(int i = start; i<end ; i++){
 double sum = 0;
 for(int j = 0; j<nCriteria; j++){
 sum += pop[i]->paretoCrowdingDistance[j];
 }
 pop[i]->avgParetoCrowdingDistance = sum/nCriteria;
 }
 }
 */


/**********************************
 NSGA III
 ********************************/

/*
 
 
 void Population::generate_recursive(int NumObj,int left, int total, int element, std::vector<double>pos, int count)
 {
 //vector<ReferencePoint> *rps, int opt.nCriteria,
 if (element == NumObj-1)
 {
 pos[element] = static_cast<double>(left)/total; // return double type left/total
 opt.rps.push_back(pos);
 //std::cout<<"size in recursive: "<<opt.rps.size()<<std::endl;
 //std::cout<<"RPS: "<< RPS[0][0]<<std::endl;
 }
 else
 {
 for (int i=0; i<=left; i+=1)
 {
 pos[element] = static_cast<double>(i)/total;
 generate_recursive(NumObj, left-i, total, element+1, pos, count);
 //std::cout<<"pos: "<<pos[element]<<std::endl;
 }
 }
 //return RPS;
 }
 
 
 void Population::GenerateReferencePoints()
 {
 opt.rps.clear();
 opt.position_.resize(3);
 opt.count = 0;
 generate_recursive(opt.nCriteria, opt.division, opt.division, 0, opt.position_, opt.count); // push back all values(from &p) in rps.........p[0] one layer (rps)
 opt.count = 0;
 opt.position_.clear();
 //std::cout<<"size of rps: "<<opt.rps.size()<<std::endl;
 //    std::cout<<"rps data: "<<" ";
 //    for(int i = 0; i<opt.rps.size(); i++){
 //        for(int j = 0; j<opt.rps[i].size(); j++){
 //            std::cout<<opt.rps[i][j]<<", ";
 //        }
 //        std::cout<<std::endl;
 //    }
 
 //////////-----------------work with only one layer only (paper specify two player for five or more objective)----------------/////////////////////////////////////////
 
 }
 
 
 //Environment Selection
 void nsga3::Population::EnvironmentalSelection(Population *pcur, int nChildren, Population *child)
 {
 //vector<size_t> considered; // St
 // ---------- Step 14 / Algorithm 2 ----------
 TranslateObjectives(pcur); // for ideal_points
 //std::cout<<"Done Translation"<<std::endl;
 
 FindExtremePoints(pcur); // for extreame_points
 //std::cout<<"Done FindExtremePoints"<<std::endl;
 
 ConstructHyperplane(); // for intercepts
 //std::cout<<"Done ConstructHyperplane"<<std::endl;
 
 NormalizeObjectives(pcur); // normalize fitness based on intercepts
 //std::cout<<"Done NormalizeObjectives"<<std::endl;
 
 // ---------- Step 15 / Algorithm 3, Step 16 ----------
 Associate(pcur); // problem in Associate
 //std::cout<<"Done Associate"<<std::endl;
 
 // ---------- Step 17 / Algorithm 4 ----------
 //std::cout<<"child_Num: "<<nChildren<<std::endl;
 
 while (nChildren < opt.popSize)
 {
 //std::cout<<"start copying from last rank till child == popSize"<<std::endl;
 int min_rp = FindNicheReferencePoint();
 //std::cout<<"min_rp from nichReferencePoint "<<min_rp<<std::endl;
 //int chosen = SelectClusterMember(opt.potential_members_, min_rp); // why -1 always
 //int chosen = SelectClusterMember(opt.potential_members_, min_rp); // why -1 always
 int chosen = SelectClusterMember(min_rp); // why -1 always
 
 //std::cout<<"chosen: "<<chosen<<std::endl;
 if (chosen < 0) // no potential member in Fl, disregard this reference point
 {
 opt.rps.erase(opt.rps.begin()+min_rp);
 //std::cout<<"member size: "<<opt.member_size_.size()<<std::endl;
 //delete all vector index with min_rp
 opt.member_size_.erase(opt.member_size_.begin()+min_rp);
 
 //works
 //opt.member_size_[min_rp] = opt.member_size_[min_rp] - 1;
 opt.potential_members_.erase(opt.potential_members_.begin() + min_rp);
 }
 else
 {
 //std::cout<<"index of rps in selection: "<<min_rp<<std::endl;
 opt.member_size_[min_rp] = opt.member_size_[min_rp] + 1;
 
 //opt.potential_members_[min_rp].erase(opt.potential_members_[min_rp].begin() + chosen);
 //remove potential member at chosen indexs
 //opt.potential_members_[min_rp][chosen][0] = -1;
 removeMember(min_rp, chosen);
 child->pop[nChildren]->copy(pop[chosen]);
 nChildren++;
 opt.childNum = nChildren;
 
 }
 //std::cout<<"number of childs: "<<nChildren<<std::endl;
 }
 
 }
 
 void nsga3::Population::TranslateObjectives(Population *p){
 
 const int NumObj = opt.nCriteria; //// nCriteria
 for (int f=0; f<NumObj; f+=1)
 {
 double minf = numeric_limits<double>::max();
 int rankZeroSize = fronts[0].size() ;//nInParetoFrontWithRank[0];
 //std::cout<<"rank 0 size: "<<fronts[0].size()<<std::endl;
 for (int i=0; i<rankZeroSize; i+=1)
 {
 minf = std::min(minf, pop[fronts[0][i]]->fitness[f]);
 }
 opt.ideal_point[f] = minf;
 //std::cout<<"f: "<<f << " "<< opt.ideal_point[f]<<std::endl;
 
 for (unsigned int t=0; t<fronts.size(); t+=1) // p-<nRanks = fornts.size()
 {
 int n = fronts[t].size(); //findFrontWithRank(t, 0, opt.popSize);
 for (int i=0; i<n; i+=1)
 {
 int ind = fronts[t][i];
 //pop[ind]->fitness[f] = pop[ind]->fitness[f] - minf;
 pop[ind]->converted_fitness[f] = pop[ind]->fitness[f] - minf; // use converted and actual fitness at different places
 }
 }
 }
 
 //return ideal_point; .............this is void function ..store all values in ideal_points[]
 }
 
 void Population::FindExtremePoints(Population *p) {
 
 for (int f=0; f<opt.nCriteria; f+=1){
 std::vector<double> w(3, 0.000001);
 w[f] = 1.0; // check again....w[f] = 0.00001 initialize
 
 double min_ASF = numeric_limits<double>::max();
 int min_indv = fronts[0].size() ;//nInParetoFrontWithRank[0]; // size of rank zero individual
 int ilast = min_indv;
 
 for (int i=0; i< fronts[0].size(); i+=1)  // only consider the individuals in the first front
 {
 double asf = ASF(opt.nCriteria, pop[ fronts[0][i] ], w); //
 
 if ( asf < min_ASF )
 {
 min_ASF = asf;
 min_indv = fronts[0][i];
 }
 }
 opt.extreme_points[f] = min_indv;
 }
 std::cout<<"extreme point: "<<opt.extreme_points[0]<<" "<<opt.extreme_points[1]<<" "<<opt.extreme_points[2]<<std::endl;
 }// FindExtremePoints()
 
 
 /////////------------ConstructHyperPlane Done -------------------------///////////////////
 void Population::ConstructHyperplane()
 {
 // Check whether there are duplicate extreme points.
 
 bool duplicate = false;
 for (int i=0; !duplicate && i<opt.nCriteria; i+=1) //
 {
 for (int j=i+1; !duplicate && j<opt.nCriteria; j+=1)
 {
 // CHECK this again
 duplicate = (opt.extreme_points[i] == opt.extreme_points[j]); // duplicate is False if all extreme points are distinct
 }
 }
 
 //-------------in SetUp function opt.intercep...............////////////////////////
 ///............................................................................/////
 
 bool negative_intercept = false;
 if (!duplicate) // if no duplicate extreme points
 {
 //std::cout<<"No dulicate points"<<std::endl;
 // Find the equation of the hyperplane
 vector<double> b(opt.nCriteria, 1.0);
 vector<double> B; //(opt.nCriteria, 1.0); //to copy in A
 //opt.b[0]= opt.b[1] = opt.b[2] = 1;
 vector< vector<double> > A;
 for (int p=0; p<opt.nCriteria; p+=1) //opt.nCriteria = extreme_points.size()
 {
 //            B.push_back(pop[ opt.extreme_points[p] ]->fitness[0]);
 //            B.push_back(pop[ opt.extreme_points[p] ]->fitness[1]);
 //            B.push_back(pop[ opt.extreme_points[p] ]->fitness[2]);
 B.push_back(pop[ opt.extreme_points[p] ]->converted_fitness[0]);
 B.push_back(pop[ opt.extreme_points[p] ]->converted_fitness[1]);
 B.push_back(pop[ opt.extreme_points[p] ]->converted_fitness[2]);
 
 A.push_back(B);
 //std::cout<<"a size: "<<A.size()<<std::endl;
 //std::cout<<pop[ opt.extreme_points[p] ]->converted_fitness[0]<<" "<<pop[ opt.extreme_points[p] ]->fitness[0]<<std::endl;
 //std::cout<<pop[ opt.extreme_points[p] ]->converted_fitness[1]<<" "<<pop[ opt.extreme_points[p] ]->fitness[1]<<std::endl;
 //std::cout<<pop[ opt.extreme_points[p] ]->converted_fitness[2]<<" "<<pop[ opt.extreme_points[p] ]->fitness[2]<<std::endl;
 B.clear();
 //std::cout<<"B size: "<<B.size()<<std::endl;
 }
 vector<double> x(3, 0);
 GuassianElimination(&x, A, b); // what this does??
 //std::cout<<"start finding intercepts"<<std::endl;
 // Find intercepts
 for (int f=0; f<opt.nCriteria; f+=1)
 {
 opt.intercepts[f] = 1.0/x[f];
 //std::cout<<"intercept : "<<f<<" "<< x[f]<<std::endl;
 
 if(x[f] < 0)
 {
 //std::cout<<"negative intercept"<<std::endl;
 negative_intercept = true;
 break;
 }
 }
 }
 
 if (duplicate || negative_intercept) //
 {
 FindMaxObjectives(); // to fins opt.max_point[f]
 for (int f=0; f<opt.nCriteria; f+=1)
 {
 opt.intercepts[f] = opt.max_point[f];
 }
 }
 }
 
 void nsga3::Population::NormalizeObjectives(Population *p){
 for (int t=0; t<fronts.size(); t+=1){ //fronts.size() = nRanks = pop->nRanks
 int rankPopSize = fronts[t].size(); // nInParetoFrontWithRank[t];
 
 for (int i=0; i<rankPopSize; i+=1) // fronts[t].size() = rankPopSize
 {
 int ind = fronts[t][i];
 for (int f=0; f<opt.nCriteria; f+=1){
 if ( fabs(opt.intercepts[f])>10e-10 ){ // avoid the divide-by-zero error
 //pop[ ind ]->fitness[f] = pop[ ind ]->fitness[f]/(opt.intercepts[f]);
 pop[ ind ]->converted_fitness[f] = pop[ ind ]->converted_fitness[f]/(opt.intercepts[f]);
 }
 else{
 //pop[ ind ]->fitness[f] = pop[ ind ]->fitness[f]/10e-10;
 pop[ ind ]->converted_fitness[f] = pop[ ind ]->converted_fitness[f]/10e-10;
 }
 
 }
 }
 }
 
 }// NormalizeObjectives()
 
 int Population::FindNicheReferencePoint()
 {
 //std::cout<<"start finding nichReferencePoint"<<std::endl;
 // find the minimal cluster size
 int min_size = numeric_limits<int>::max(); // use int instead of size_t
 for (unsigned int r=0; r<opt.rps.size(); r+=1)
 {
 //std::cout<<"min: "<<min_size<<" "<<"r: "<< opt.member_size_[r]<<" "<<opt.rps.size()<<std::endl;
 min_size = std::min(min_size, opt.member_size_[r]);
 //std::cout<<"member at r: "<<r<<" "<<opt.member_size_[r]<<std::endl;
 }
 
 // find the reference points with the minimal cluster size Jmin
 vector<int> min_rps;
 for (unsigned int r=0; r<opt.rps.size(); r+=1)
 {
 if (opt.member_size_[r] == min_size)
 {
 min_rps.push_back(r);
 }
 }
 //std::cout<<"size of min_rps: "<<min_rps.size()<<std::endl;
 // return a random reference point (j-bar)
 //std::cout<<min_rps[rand()%min_rps.size()]<<std::endl;
 return min_rps[rand()%min_rps.size()];
 }
 
 int nsga3::Population::SelectClusterMember(int index) // ms= member size
 {
 int chosen = -1;
 int x =  opt.potential_members_[index].vect.size();
 //std::cout<<"x: "<<x<<std::endl;
 if (x != 0)
 {
 //std::cout<<"index: "<<index<<" "<<opt.member_size_[index]<<std::endl;
 if (opt.member_size_[index] == 0) // currently has no member
 {
 //std::cout<<"chose closest"<<std::endl;
 chosen =  FindClosestMember(index);
 }
 else
 {
 //std::cout<<"chose randomly"<<std::endl;
 chosen =  RandomMember();
 }
 }
 return chosen;
 }
 
 
 void Population::Associate(Population* p){
 //std::cout<<"Start Association"<<std::endl;
 //std::cout<<"size of potential member1: "<<opt.potential_members_.size()<<std::endl;
 std::vector<double> direction;
 std::vector<double> tmpfit;
 std::vector<double> dist(1);
 std::vector<double> individual(1);
 std::vector<std::vector<double>> ind(1);
 //std::cout<<"ref size:"<<opt.rps.size()<<std::endl;
 opt.member_size_.clear();
 opt.member_size_.resize(opt.rps.size());
 opt.potential_members_.clear();
 opt.potential_members_.resize(opt.rps.size());
 //std::cout<<"ms_size: "<<opt.member_size_.size()<<std::endl;
 fillMembers(opt.member_size_.size());
 //std::cout<<"all members are initialize in Associate"<<std::endl;
 
 for (unsigned int t=0; t<fronts.size(); t+=1) // how many front are there...how many rank (nRanks) .... pop->nRanks
 {
 //std::cout<<"fronts size: "<<fronts.size()<<std::endl;
 int n =  fronts[t].size() ;//nInParetoFrontWithRank[t];
 //std::cout<<"rank size "<<t<<" "<<n<<std::endl;
 for (int i=0; i<n; i+=1) // how many individual in each rank
 {
 int min_rp = opt.rps.size(); // size of reference point vector
 min_rp = min_rp - 1;
 //std::cout<<"iiiiiii: "<<min_rp<<std::endl;
 double min_dist = numeric_limits<double>::max();
 for (unsigned int r=0; r<opt.rps.size(); r+=1)
 {
 //std::cout<<"find perpendicular dist " <<opt.rps[r][2]<<std::endl;
 direction.push_back(opt.rps[r][0]); // rps is 2d vector
 direction.push_back(opt.rps[r][1]);
 direction.push_back(opt.rps[r][2]);
 tmpfit.push_back(p->pop[fronts[t][i]]->converted_fitness[0]);
 tmpfit.push_back(p->pop[fronts[t][i]]->converted_fitness[1]);
 tmpfit.push_back(p->pop[fronts[t][i]]->converted_fitness[2]);
 //std::cout<<"fit0: "<<p->pop[fronts[t][i]]->fitness[0]<<std::endl;
 //std::cout<<"start prependicular calculation"<<std::endl;
 double d = PerpendicularDistance(direction, tmpfit); //Look into D calculation
 //std::cout<<"d: "<<d<<std::endl;
 if (d < min_dist)
 {
 min_dist = d;
 min_rp = r;
 }
 //we need to clear direction and tmpfit vector to push again
 direction.clear();
 tmpfit.clear();
 //std::cout<<"tmpfit size: "<<tmpfit.size()<<std::endl;
 }
 
 if (t+1 != fronts.size()) // associating members in St/Fl (only counting)
 {
 opt.member_size_[min_rp] = opt.member_size_[min_rp] + 1; //
 //std::cout<<"Fl only  "<<"min_rp"<<min_rp<<" "<<opt.member_size_[min_rp]<<std::endl;
 }
 else
 {
 opt.potential_members_[min_rp].vect.push_back(make_pair(fronts[t][i], min_dist));
 dist.clear();
 individual.clear();
 ind.clear();
 //std::cout<<"Associate DOne"<<std::endl;
 }
 
 }// for - members in front
 }// for - fronts
 //std::cout<<"size of potential member: "<<opt.potential_members_.size()<<std::endl;
 }
 
 
 */

void clear_teams(vector<population>* teams){
    for (int team_number = 0 ; team_number < teams->size(); team_number++) {
        for (int rover = 0 ; rover < teams->at(team_number).teamRover.size(); rover++) {
            for (int neural = 0 ; neural < teams->at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).right_rover.clear();
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).left_rover.clear();
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).target_distance.clear();
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).obstacle_distance.clear();
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.clear();
                teams->at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.clear();
            }
        }
    }
}

void run_simulation_function(){
    
    int number_of_teams=1;
    int number_of_rover = 3;
    int number_of_routes = 8;
    double distance_between_rover = 3;
    double safe_distance_between_rover = 0.5;
    int number_of_obstacles = 3;
    double radius_of_obstacle = 2.0;
    
    //Create teams
    vector<population> teams;
    vector<population>* p_teams = &teams;
    for (int team_number = 0 ; team_number < number_of_teams; team_number++) {
        population p(number_of_rover, number_of_routes);
        p_teams->push_back(p);
    }
    
    //This is to make sure each agent starts at the location decided
    vector<vector<double>> coordinates_stat;
    vector<vector<double>>* p_coordinates_stat = &coordinates_stat;
    for (int rover = 0 ; rover <number_of_rover; rover++) {
        vector<double> temp;
        double temp_x_const = 2.0+(distance_between_rover*rover);
        double temp_y_const = 0.0;
        temp.push_back(temp_x_const);
        temp.push_back(temp_y_const);
        coordinates_stat.push_back(temp);
        temp.clear();
    }
    
    //Obstacles
    vector<vector<double>> location_obstacle;
    vector<vector<double>>* p_location_obstacle = &location_obstacle;
    
    
    int number_of_generations = 500;

    for (int generation = 0 ; generation < number_of_generations; generation++) {
        cout<<generation<<endl;
        initial_team(p_teams, p_location_obstacle,number_of_obstacles,p_coordinates_stat, distance_between_rover);
        simulation_team(p_teams, p_location_obstacle, generation,number_of_obstacles,p_coordinates_stat, distance_between_rover);
        distance_team(p_teams, distance_between_rover, safe_distance_between_rover, radius_of_obstacle, p_location_obstacle);
        reward_team(p_teams, distance_between_rover, safe_distance_between_rover, radius_of_obstacle,number_of_obstacles);
        
        if (generation == (number_of_generations-1)) {
            FILE* p_xy;
            p_xy = fopen("XY", "a");
            for (int team_number = 0 ; team_number < teams.size(); team_number++) {
                for (int rover = 0 ; rover < teams.at(team_number).teamRover.size(); rover++) {
                    for (int neural = 0 ; neural < teams.at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                        for (int index = 0; index < teams.at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.size(); index++) {
                            fprintf(p_xy, "%f \t %f \n",teams.at(team_number).teamRover.at(rover).new_network.at(neural).x_coordinates.at(index),teams.at(team_number).teamRover.at(rover).new_network.at(neural).y_coordinates.at(index));
                        }
                        fprintf(p_xy, "\n");
                    }
                    fprintf(p_xy, "\n");
                }
                fprintf(p_xy, "\n");
            }
            fclose(p_xy);
            
            
            FILE* p_rewards;
            p_rewards = fopen("rewards", "a");
            for (int team_number = 0 ; team_number < teams.size(); team_number++) {
                for (int rover = 0 ; rover < teams.at(team_number).teamRover.size(); rover++) {
                    for (int neural = 0; neural < teams.at(team_number).teamRover.at(rover).new_network.size(); neural++) {
                        fprintf(p_rewards, "%d \t %d \t %d \t %f \t %f \t %f \n",team_number,rover,neural, teams.at(team_number).teamRover.at(rover).new_network.at(neural).hitting_obstacle,teams.at(team_number).teamRover.at(rover).new_network.at(neural).shortest_target_distance,teams.at(team_number).teamRover.at(rover).new_network.at(neural).hitting_other_rover);
                        cout<<teams.at(team_number).teamRover.at(rover).new_network.at(neural).shortest_target_distance<<endl;
                    }
                }
            }
            fclose(p_rewards);
        }
        
        
        ea(p_teams);
        clear_teams(p_teams);

    }
}

int main(int argc, const char * argv[]) {
    srand ( time(NULL) );
    run_simulation_function();
    return 0;
}


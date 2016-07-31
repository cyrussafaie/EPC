#include "psyho_sol.h"

#include <chrono>

std::vector<std::string> split(const std::string& s, char delim)
{
	std::vector<std::string> elems;
	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, delim))
	{
		elems.push_back(item);
	}
	return elems;
}

std::string trim(const std::string& s)
{
	auto nonSpace = [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); };
	auto begin = std::find_if(s.begin(), s.end(), nonSpace);
	if (begin == s.end())
		return std::string();
	auto end = std::find_if(s.rbegin(), s.rend(), nonSpace).base();
	return std::string(begin, end);
}

std::vector<std::string> read_data(const std::string& filename, bool skipHeader = false)
{
    std::vector<std::string> data;
    std::ifstream ifs(filename);
    std::string line;
    while(std::getline(ifs, line))
    {
        if(skipHeader)
            skipHeader = false;
        else
            data.push_back(trim(line));
    }
    return data;
}

double score_results(std::vector<std::string> result, std::map<int, int>& groundTruthA, std::map<int, int>& groundTruthB, std::map<int, int> transactionCountsA, std::map<int, int> transactionCountsB)
{
    std::vector<std::vector<double>> lossMatrix = {
        {0.00, 0.20, 0.70},
        {0.50, 0.00, 0.01},
        {1.00, 0.01, 0.00}
    };
    
    int numRA = 0, numRB = 0;
    int numGA = 0, numGB = 0;
    
    double cost = 0, fullCost = 0;
    
    std::vector<std::vector<int>> confusionMatrixA(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> confusionMatrixB(3, std::vector<int>(3, 0));

    std::vector<std::vector<int>> confusionMatrixA2(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> confusionMatrixB2(3, std::vector<int>(3, 0));

    std::vector<std::vector<std::set<int>>> IDsA(3, std::vector<std::set<int>>(3));
    std::vector<std::vector<std::set<int>>> IDsB(3, std::vector<std::set<int>>(3));

    for(int i = 0; i < result.size(); i++)
    {
        std::vector<std::string> tokens = split(result[i], ',');
        
        int id = atoi(tokens[0].c_str());
        
        int gtA = -1;
        if(groundTruthA.find(id) != groundTruthA.end())
            gtA = groundTruthA[id];
        
        int gtB = -1;
        if(groundTruthB.find(id) != groundTruthB.end())
            gtB = groundTruthB[id];
        
        if(gtA != -1)
            numGA++;
        if(gtB != -1)
            numGB++;
        
        int rA = -1;
        if(tokens[1] == "No")
            rA = 0;
        else if(tokens[1] == "Maybe")
            rA = 1;
        else if(tokens[1] == "Yes")
            rA = 2;
        
        int rB = -1;
        if(tokens[2] == "No")
            rB = 0;
        else if(tokens[2] == "Maybe")
            rB = 1;
        else if(tokens[2] == "Yes")
            rB = 2;
        
        if(rA != -1)
            numRA++;
        if(rB != -1)
            numRB++;
        
        if((rA == -1) != (gtA == -1))
            std::cout << "Bad A: " << rA << ", " << gtA << "\n";
        if((rB == -1) != (gtB == -1))
            std::cout << "Bad B: " << rB << ", " << gtB << "\n";
        
        if(gtA != -1)
        {
            confusionMatrixA[rA][gtA]++;
            confusionMatrixA2[rA][gtA] += transactionCountsA[id];
            IDsA[rA][gtA].insert(id);

            cost += lossMatrix[rA][gtA];
            fullCost++;
        }
        if(gtB != -1)
        {
            confusionMatrixB[rB][gtB]++;
            confusionMatrixB2[rB][gtB] += transactionCountsB[id];
            IDsB[rB][gtB].insert(id);

            cost += lossMatrix[rB][gtB];
            fullCost++;
        }
    }

    std::cout << "confusionMatrixA:\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::setw(5) << confusionMatrixA[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "confusionMatrixB:\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::setw(5) << confusionMatrixB[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "confusionMatrixA2:\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::setw(8) << confusionMatrixA2[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "confusionMatrixB2:\n";
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << std::setw(8) << confusionMatrixB2[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::string names[] = {"No", "Maybe", "Yes"};

    std::ofstream ofs2("out_CM.csv");
    ofs2 << "Confusion Matrix Element,Segment,Item ID,Ground Truth,Prediction\n";
    for(int i = 0; i < 9; i++)
    {
        const int r = i / 3;
        const int gt = i % 3;
        for(int id : IDsA[r][gt])
        {
            ofs2 << (char)('a' + i) << ",A," << id << "," << names[gt] << "," << names[r] << "\n";
        }
        for(int id : IDsB[r][gt])
        {
            ofs2 << (char)('a' + i) << ",B," << id << "," << names[gt] << "," << names[r] << "\n";
        }
    }
    ofs2.close();

    std::ofstream ofs3("out_CM_2.csv");
    ofs3 << "Confusion Matrix Element,Segment,Item ID,Transaction Counts,Ground Truth,Prediction\n";
    for(int i = 0; i < 9; i++)
    {
        const int r = i / 3;
        const int gt = i % 3;
        for(int id : IDsA[r][gt])
        {
            ofs3 << (char)('a' + i) << ",A," << id << "," << transactionCountsA[id] << "," << names[gt] << "," << names[r] << "\n";
        }
        for(int id : IDsB[r][gt])
        {
            ofs3 << (char)('a' + i) << ",B," << id << "," << transactionCountsB[id] << "," << names[gt] << "," << names[r] << "\n";
        }
    }
    ofs3.close();
    
    return (1 - cost / fullCost) * 1e6;
}

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		std::cerr << "Usage: " << argv[0] << " <training data> <testing data>" << std::endl;
		return 1;
	}

    std::vector<std::string> trainData = read_data(argv[1], true);
	std::vector<std::string> testData = read_data(argv[2], true);

    std::map<int, int> transactionCountsA, transactionCountsB;

    std::map<int, int> groundTruthA, groundTruthB;
    for(int i = 0; i < testData.size(); i++)
    {
        std::vector<std::string> tokens = split(testData[i], ',');
        
        int id = atoi(tokens[0].c_str());
        
        int gt = -1;
        if(tokens[28] == "No")
            gt = 0;
        else if(tokens[28] == "Maybe")
            gt = 1;
        else if(tokens[28] == "Yes")
            gt = 2;
        
        if(tokens[8][0] == 'A')
        {
            groundTruthA[id] = gt;
            transactionCountsA[id]++;
        }
        else if(tokens[8][0] == 'B')
        {
            groundTruthB[id] = gt;
            transactionCountsB[id]++;
        }
    }

    std::default_random_engine rng(12345);
    
    std::vector<std::string> trainingData, testingData, result;
    
    for(int i = 0; i < trainData.size(); i++)
    {
        trainingData.push_back(trainData[i]);
    }
    for(int i = 0; i < testData.size(); i++)
    {
        testingData.push_back(testData[i].substr(0, testData[i].find_last_of(',')));
    }
	
   	auto t1 = std::chrono::high_resolution_clock::now();
     	
  	ElectronicPartsClassification epc;
  	result = epc.classifyParts(trainingData, testingData, std::vector<int>());
  	
   	auto t2 = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = t2 - t1;
  	
  	double score = score_results(result, groundTruthA, groundTruthB, transactionCountsA, transactionCountsB);

    std::cout << "Score: " << score << "\n";

    std::ofstream ofs_res("out_result.txt");
    for(int i = 0; i < result.size(); i++)
        ofs_res << result[i] << "\n";
    ofs_res.close();
}

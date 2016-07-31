// Author: Psyho
// Blog: http://psyho.gg/
// Twitter: https://twitter.com/fakepsyho


const int TRAIN_AUGMENTS = 200;
const int TEST_AUGMENTS = 200;
const bool USE_CUSTOMER_PREDICTIONS = false;
const bool USE_BASIC_FEATURES = true;



#include <bits/stdc++.h>
#include <sys/time.h>
//#include <emmintrin.h>

int THREADS_NO = 4;

using namespace std;

#define INLINE   inline __attribute__ ((always_inline))
#define NOINLINE __attribute__ ((noinline))

#define ALIGNED __attribute__ ((aligned(16)))

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#define SSELOAD(a)     _mm_load_si128((__m128i*)&a)
#define SSESTORE(a, b) _mm_store_si128((__m128i*)&a, b)

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define REP(i,a)    FOR(i,0,a)
#define ZERO(m)     memset(m,0,sizeof(m))
#define ALL(x)      x.begin(),x.end()
#define PB          push_back
#define S           size()
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define X           first
#define Y           second
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VVVD        VC < VVD >
#define VS          VC < string >
#define VVS         VC < VS >
#define DB(a)       cerr << #a << ": " << (a) << endl;
#define PDD pair<double, double>
#define VPDD VC<PDD>
#define VVPDD VC<VPDD>

template<class A, class B> ostream& operator<<(ostream &os, pair<A,B> &p) {os << "(" << p.X << "," << p.Y << ")"; return os;}
template<class A, class B, class C> ostream& operator<<(ostream &os, tuple<A,B,C> &p) {os << "(" << get<0>(p) << "," << get<1>(p) << "," << get<2>(p) << ")"; return os;}
template<class T> ostream& operator<<(ostream &os, VC<T> &v) {os << "{"; REP(i, v.S) {if (i) os << ", "; os << v[i];} os << "}"; return os;}
template<class T> ostream& operator<<(ostream &os, set<T> &s) {VS vs(ALL(s)); return os << vs;}
template<class T> string i2s(T x) {ostringstream o; o << x; return o.str();}
VS splt(string s, char c = ' ') {VS all; int p = 0, np; while (np = s.find(c, p), np >= 0) {all.PB(s.substr(p, np - p)); p = np + 1;} all.PB(s.substr(p)); return all;}

double getTime() {
	timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct RNG {
    unsigned int MT[624];
    int index;
	
	RNG(int seed = 1) {
		init(seed);
	}
    
    void init(int seed = 1) {
        MT[0] = seed;
        FOR(i, 1, 624) MT[i] = (1812433253UL * (MT[i-1] ^ (MT[i-1] >> 30)) + i);
        index = 0;
    }
    
    void generate() {
        const unsigned int MULT[] = {0, 2567483615UL};
        REP(i, 227) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i+397] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        FOR(i, 227, 623) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i-227] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        unsigned int y = (MT[623] & 0x8000000UL) + (MT[0] & 0x7FFFFFFFUL);
        MT[623] = MT[623-227] ^ (y >> 1);
        MT[623] ^= MULT[y&1];
    }
    
    unsigned int rand() {
        if (index == 0) {
            generate();
        }
        
        unsigned int y = MT[index];
        y ^= y >> 11;
        y ^= y << 7  & 2636928640UL;
        y ^= y << 15 & 4022730752UL;
        y ^= y >> 18;
        index = index == 623 ? 0 : index + 1;
        return y;
    }
    
    INLINE int next() {
        return rand();
    }
    
    INLINE int next(int x) {
        return rand() % x;
    }
    
    INLINE int next(int a, int b) {
        return a + (rand() % (b - a));
    }
    
    INLINE double nextDouble() {
        return (rand() + 0.5) * (1.0 / 4294967296.0);
    }
};

struct TreeNode {
	int level;
	int feature;
	double value;
	double result;
	int left;
	int right;
	
	TreeNode() {
		level = -1;
		feature = -1;
		value = 0;
		result = 0;
		left = -1;
		right = -1;
	}
};

struct MLConfig {
	static const int MSE = 0;
	static const int MCE = 1;
	static const int MAE = 2;
	static const int MQE = 3;
	static const int LOG = 4;
	static const int CUSTOM = 5;

	VI randomFeatures = {5};
	VI randomPositions = {2};
	int featuresIgnored = 0;
	int groupFeature = -1;
	VI groups = {};
	int maxLevel = 100;
	int maxNodeSize = 1;
	int maxNodes = 0;
	int threadsNo = 1;
	int treesNo = 1;
	double minSplitValue = 1e99;
	double regularization = 0.0;
	double bagSize = 1.0;
	double timeLimit = 0;
	int lossFunction = MSE;
	bool useBootstrapping = true;
	bool computeImportances = false; 
	bool saveChosenSamples = false;
	
	//Boosting
	bool useLineSearch = false;
	double shrinkage = 0.1;
};

class DecisionTree {
	public:
	VC<TreeNode> nodes;
	VD importances;
	VI samplesChosen;
	
	private:
	template <class T> INLINE T customLoss(T x) {
		// const double ALPHA = .25;
		// return abs(x) < ALPHA ? x * x / 2.0 : ALPHA * (abs(x) - ALPHA / 2);
		return abs(x) * sqrt(abs(x));
	}
	
	
	public:
	
	DecisionTree() { }
	
	template <class T> INLINE DecisionTree(VC<VC<T>> &features, VC<T> &results, MLConfig &config, int seed, const int USE_WEIGHTS, const int SCORING_FUNCTION) {
		RNG r(seed);
		
		if (config.computeImportances) {
			importances = VD(features[0].S);
		}
		
		VI chosenGroups(features.S);
		if (config.groupFeature == -1 && config.groups.S == 0) {
			REP(i, (int)(features.S * config.bagSize)) chosenGroups[r.next(features.S)]++;
		} else if (config.groupFeature != -1) {
			assert(config.groupFeature >= 0 && config.groupFeature < features.S);
			unordered_map<T, int> groups;
			int groupsNo = 0;
			REP(i, features.S) if (!groups.count(features[i][config.groupFeature])) {
				groups[features[i][config.groupFeature]] = groupsNo++;
			}
			VI groupSize(groupsNo);
			REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
			REP(i, features.S) chosenGroups[i] = groupSize[groups[features[i][config.groupFeature]]];
		} else {
			assert(config.groups.S == features.S);
			int groupsNo = 0;
			for (int x : config.groups) groupsNo = max(groupsNo, x + 1);
			VI groupSize(groupsNo);
			REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
			REP(i, features.S) chosenGroups[i] = groupSize[config.groups[i]];
		}
		
		int bagSize = 0;
		REP(i, features.S) if (chosenGroups[i]) bagSize++;
		
		VI bag(bagSize);
		VI weight(features.S);
		
		int pos = 0;
		
		REP(i, features.S) {
			weight[i] = config.useBootstrapping ? chosenGroups[i] : min(1, chosenGroups[i]);
			if (chosenGroups[i]) bag[pos++] = i;
		}
		
		if (config.saveChosenSamples) samplesChosen = chosenGroups;
		
		TreeNode root;
		root.level = 0;
		root.left = 0;
		root.right = pos;
		nodes.PB(root);
		
		queue<int> stack;
		stack.push(0);
		
		while (!stack.empty()) {
			bool equal = true;
			
			int curNode = stack.front(); stack.pop();
			
			int bLeft = nodes[curNode].left;
			int bRight = nodes[curNode].right;
			int bSize = bRight - bLeft;
			
			int totalWeight = 0; 
			T totalSum = 0;
			T total2Sum = 0;
			FOR(i, bLeft, bRight) {
				if (USE_WEIGHTS) {
					totalSum += results[bag[i]] * weight[bag[i]];
					totalWeight += weight[bag[i]];
					if (SCORING_FUNCTION == MLConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]] * weight[bag[i]];
				} else {
					totalSum += results[bag[i]];
					if (SCORING_FUNCTION == MLConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]];
				}
			}			
			assert(bSize > 0);
			
			if (!USE_WEIGHTS) totalWeight = bSize;
			
			FOR(i, bLeft + 1, bRight) if (results[bag[i]] != results[bag[i - 1]]) {
				equal = false;
				break;
			}
			
			if (equal || bSize <= config.maxNodeSize || nodes[curNode].level >= config.maxLevel || config.maxNodes && nodes.S >= config.maxNodes) {
				if (SCORING_FUNCTION == MLConfig::LOG) {
					double s1 = 0, s2 = 0;
					FOR(i, bLeft, bRight) {
						if (USE_WEIGHTS) {
							s1 += results[bag[i]] * weight[bag[i]];
							s2 += results[features.S + bag[i]] * weight[bag[i]];
						} else {
							s1 += results[bag[i]];
							s2 += results[features.S + bag[i]];
						}
					}			
					nodes[curNode].result = -s1 / s2;
				} else {
					nodes[curNode].result = totalSum / totalWeight;
				}
				continue;
			}
			
			int bestFeature = -1;
			int bestLeft = 0;
			int bestRight = 0;
			T bestValue = 0;
			T bestLoss = 1e99;
			
			const int randomFeatures = config.randomFeatures[min(nodes[curNode].level, (int)config.randomFeatures.S - 1)];
			REP(i, randomFeatures) {
			
				int featureID = config.featuresIgnored + r.next(features[0].S - config.featuresIgnored);
				
				T vlo, vhi;
				vlo = vhi = features[bag[bLeft]][featureID];
				FOR(j, bLeft + 1, bRight) {
					vlo = min(vlo, features[bag[j]][featureID]);
					vhi = max(vhi, features[bag[j]][featureID]);
				}
				if (vlo == vhi) continue;
				
				const int randomPositions = config.randomPositions[min(nodes[curNode].level, (int)config.randomPositions.S - 1)];
				REP(j, randomPositions) {
					T splitValue = features[bag[bLeft + r.next(bSize)]][featureID];
					if (splitValue == vlo) {
						j--;
						continue;
					}
					
					if (SCORING_FUNCTION == MLConfig::MSE) {
						T sumLeft = 0;
						T sum2Left = 0;
						int totalLeft = 0;
						FOR(k, bLeft, bRight) {
							int p = bag[k];
							if (features[p][featureID] < splitValue) {
								if (USE_WEIGHTS) {
									sumLeft += results[p] * weight[p];
									sum2Left += results[p] * results[p] * weight[p];
								} else {
									sumLeft += results[p];
									sum2Left += results[p] * results[p];
								}
								totalLeft++;
							}
						}
						
						T sumRight = totalSum - sumLeft;
						T sum2Right = total2Sum - sum2Left;
						int totalRight = bSize - totalLeft;
						
						if (totalLeft == 0 || totalRight == 0)
							continue;
						
						T loss = sum2Left - sumLeft * sumLeft / totalLeft + sum2Right - sumRight * sumRight / totalRight;
						
						if (loss < bestLoss) {
							bestLoss = loss;
							bestValue = splitValue;
							bestFeature = featureID;
							bestLeft = totalLeft;
							bestRight = totalRight;
							if (loss == 0) goto outer;
						}
					} else {
						T sumLeft = 0;
						int totalLeft = 0;
						int weightLeft = 0;
						FOR(k, bLeft, bRight) {
							int p = bag[k];
							if (features[p][featureID] < splitValue) {
								if (USE_WEIGHTS) {
									sumLeft += results[p] * weight[p];
									weightLeft += weight[p];
								} else {
									sumLeft += results[p];
								}
								totalLeft++;
							}
						}
						
						if (!USE_WEIGHTS) weightLeft = totalLeft;
						
						T sumRight = totalSum - sumLeft;
						int weightRight = totalWeight - weightLeft;
						int totalRight = bSize - totalLeft;
						
						if (totalLeft == 0 || totalRight == 0)
							continue;
						
						T meanLeft = sumLeft / weightLeft;
						T meanRight = sumRight / weightRight;
						T loss = 0;
						
						if (SCORING_FUNCTION == MLConfig::MCE) {
							FOR(k, bLeft, bRight) {
								int p = bag[k];
								if (features[p][featureID] < splitValue) {
									loss += abs(results[p] - meanLeft)  * (results[p] - meanLeft)  * (results[p] - meanLeft)  * weight[p];
								} else {
									loss += abs(results[p] - meanRight) * (results[p] - meanRight) * (results[p] - meanRight) * weight[p];
								}
								if (loss > bestLoss) break; //OPTIONAL
							}
						} else if (SCORING_FUNCTION == MLConfig::MQE) {
							FOR(k, bLeft, bRight) {
								int p = bag[k];
								if (features[p][featureID] < splitValue) {
									loss += (results[p] - meanLeft)  * (results[p] - meanLeft)  * (results[p] - meanLeft)  * (results[p] - meanLeft)  * weight[p];
								} else {
									loss += (results[p] - meanRight) * (results[p] - meanRight) * (results[p] - meanRight) * (results[p] - meanRight) * weight[p];
								}
								if (loss > bestLoss) break; //OPTIONAL
							}
						} else if (SCORING_FUNCTION == MLConfig::MAE) {
							FOR(k, bLeft, bRight) {
								int p = bag[k];
								if (features[p][featureID] < splitValue) {
									loss += abs(results[p] - meanLeft)  * weight[p];
								} else {
									loss += abs(results[p] - meanRight) * weight[p];
								}
								if (loss > bestLoss) break; //OPTIONAL
							}
						} else if (SCORING_FUNCTION == MLConfig::LOG) {
							double l1 = 0, r1 = 0;
							double l2 = 0, r2 = 0;
							FOR(k, bLeft, bRight) {
								int p = bag[k];
								if (features[p][featureID] < splitValue) {
									l1 += results[p] * weight[p];
									l2 += results[features.S + p] * weight[p];
								} else {
									r1 += results[p] * weight[p];
									r2 += results[features.S + p] * weight[p];
								}
							}
							loss = l1 * l1 / (l2 + config.regularization) + r1 * r1 / (r2 + config.regularization) - (l1 + r1) * (l1 + r1) / (l2 + r2 + config.regularization);
							loss = -loss;
						} else if (SCORING_FUNCTION == MLConfig::CUSTOM) {
							FOR(k, bLeft, bRight) {
								int p = bag[k];
								if (features[p][featureID] < splitValue) {
									loss += customLoss(results[p] - meanLeft)  * weight[p];
								} else {
									loss += customLoss(results[p] - meanRight) * weight[p];
								}
								if (loss > bestLoss) break; //OPTIONAL
							}
						}
						
						if (loss < bestLoss) {
							bestLoss = loss;
							bestValue = splitValue;
							bestFeature = featureID;
							bestLeft = totalLeft;
							bestRight = totalRight;
							if (loss == 0) goto outer;
						}
					}
				}
			}
			outer: 
			
			if (bestLoss > config.minSplitValue || bestLeft == 0 || bestRight == 0) {
				if (SCORING_FUNCTION == MLConfig::LOG) {
					double s1 = 0, s2 = 0;
					FOR(i, bLeft, bRight) {
						if (USE_WEIGHTS) {
							s1 += results[bag[i]] * weight[bag[i]];
							s2 += results[features.S + bag[i]] * weight[bag[i]];
						} else {
							s1 += results[bag[i]];
							s2 += results[features.S + bag[i]];
						}
					}			
					nodes[curNode].result = -s1 / s2;
				} else {
					nodes[curNode].result = totalSum / totalWeight;
				}
				continue;
			}
			
			if (config.computeImportances) {
				importances[bestFeature] += bRight - bLeft;
			}
			
			T mean = totalSum / totalWeight;
			
			T nextValue = -1e99;
			FOR(i, bLeft, bRight) if (features[bag[i]][bestFeature] < bestValue) nextValue = max(nextValue, features[bag[i]][bestFeature]);
			
			TreeNode left;
			TreeNode right;
			
			left.level = right.level = nodes[curNode].level + 1;
			nodes[curNode].feature = bestFeature;
			nodes[curNode].value = (bestValue + nextValue) / 2.0;
			if (!(nodes[curNode].value > nextValue)) nodes[curNode].value = bestValue;
			nodes[curNode].left = nodes.S;
			nodes[curNode].right = nodes.S + 1;
			
			int bMiddle = bRight;
			FOR(i, bLeft, bMiddle) {
				if (features[bag[i]][nodes[curNode].feature] >= nodes[curNode].value) {
					swap(bag[i], bag[--bMiddle]);
					i--;
					continue;
				}
			}
			
			assert(bestLeft == bMiddle - bLeft);
			assert(bestRight == bRight - bMiddle);
			
			left.left = bLeft;
			left.right = bMiddle;
			right.left = bMiddle;
			right.right = bRight;
			
			stack.push(nodes.S);
			stack.push(nodes.S + 1);
			
			nodes.PB(left);
			nodes.PB(right);
			
		}
		
		nodes.shrink_to_fit();
	}
	
	template <class T> double predict(VC<T> &features) {
		TreeNode *pNode = &nodes[0];
		while (true) {
			if (pNode->feature < 0) return pNode->result;
			pNode = &nodes[features[pNode->feature] < pNode->value ? pNode->left : pNode->right];
		}
	}
};

RNG gRNG(1);

class TreeEnsemble {
	public:

	VC<DecisionTree> trees;
	VD importances;
	MLConfig config;
	
	void clear() {
		trees.clear();
		trees.shrink_to_fit();
	}
	
	template <class T> DecisionTree createTree(VC<VC<T>> &features, VC<T> &results, MLConfig &config, int seed) {
		if (config.useBootstrapping) {
			if (config.lossFunction == MLConfig::MAE) {
				return DecisionTree(features, results, config, seed, true, MLConfig::MAE);
			} else if (config.lossFunction == MLConfig::MSE) {
				return DecisionTree(features, results, config, seed, true, MLConfig::MSE);
			} else if (config.lossFunction == MLConfig::MQE) {
				return DecisionTree(features, results, config, seed, true, MLConfig::MQE);
			} else if (config.lossFunction == MLConfig::MCE) {
				return DecisionTree(features, results, config, seed, true, MLConfig::MCE);
			} else if (config.lossFunction == MLConfig::LOG) {
				return DecisionTree(features, results, config, seed, true, MLConfig::LOG);
			} else if (config.lossFunction == MLConfig::CUSTOM) {
				return DecisionTree(features, results, config, seed, true, MLConfig::CUSTOM);
			}
		} else {
			if (config.lossFunction == MLConfig::MAE) {
				return DecisionTree(features, results, config, seed, false, MLConfig::MAE);
			} else if (config.lossFunction == MLConfig::MSE) {
				return DecisionTree(features, results, config, seed, false, MLConfig::MSE);
			} else if (config.lossFunction == MLConfig::MCE) {
				return DecisionTree(features, results, config, seed, false, MLConfig::MCE);
			} else if (config.lossFunction == MLConfig::MQE) {
				return DecisionTree(features, results, config, seed, false, MLConfig::MQE);
			} else if (config.lossFunction == MLConfig::LOG) {
				return DecisionTree(features, results, config, seed, false, MLConfig::LOG);
			} else if (config.lossFunction == MLConfig::CUSTOM) {
				return DecisionTree(features, results, config, seed, false, MLConfig::CUSTOM);
			}
		}
	}
	
	LL countTotalNodes() {
		LL rv = 0;
		REP(i, trees.S) rv += trees[i].nodes.S;
		return rv;
	}
	
	void printImportances() {
		assert(config.computeImportances);
		
		VC<pair<double, int>> vp;
		REP(i, importances.S) vp.PB(MP(importances[i], i));
		sort(vp.rbegin(), vp.rend());
		
		// REP(i, importances.S) printf("#%02d: %.6lf\n", vp[i].Y, vp[i].X);
		REP(i, importances.S) printf("%d, ", vp[i].Y);
	}
	
};

class RandomForest : public TreeEnsemble {
	public:
	
	template <class T> void train(VC<VC<T>> &features, VC<T> &results, MLConfig &_config, int treesMultiplier = 1) {
		double startTime = getTime();
		config = _config;
		
		int treesNo = treesMultiplier * config.treesNo;
		
		if (config.threadsNo == 1) {
			REP(i, treesNo) {	
				if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
				trees.PB(createTree(features, results, config, gRNG.next()));
			}
		} else {
			thread *threads = new thread[config.threadsNo];
			mutex mutex;
			REP(i, config.threadsNo) 
				threads[i] = thread([&] {
					while (true) {
						mutex.lock();
						int seed = gRNG.next();
						mutex.unlock();
						auto tree = createTree(features, results, config, seed);
						mutex.lock();
						if (trees.S < treesNo)
							trees.PB(tree);
						bool done = trees.S >= treesNo || config.timeLimit && getTime() - startTime > config.timeLimit;
						mutex.unlock();
						if (done) break;
					}
				});
			REP(i, config.threadsNo) threads[i].join();
			delete[] threads;
		}
		
		if (config.computeImportances) {
			importances = VD(features[0].S);
			for (DecisionTree tree : trees)
				REP(i, importances.S)
					importances[i] += tree.importances[i];
			double sum = 0;
			REP(i, importances.S) sum += importances[i];
			REP(i, importances.S) importances[i] /= sum;
		}
	}
	
	template <class T> double predict(VC<T> &features) {
		assert(trees.S);
	
		double sum = 0;
		REP(i, trees.S) sum += trees[i].predict(features);
		return sum / trees.S;
	}
	
	template <class T> VD predict(VC<VC<T>> &features) {
		assert(trees.S);
	
		int samplesNo = features.S;
	
		VD rv(samplesNo);
		if (config.threadsNo == 1) {
			REP(j, samplesNo) {
				REP(i, trees.S) rv[j] += trees[i].predict(features[j]);
				rv[j] /= trees.S;
			}
		} else {
			thread *threads = new thread[config.threadsNo];
			REP(i, config.threadsNo) 
				threads[i] = thread([&](int offset) {
					for (int j = offset; j < samplesNo; j += config.threadsNo) {
						REP(k, trees.S) rv[j] += trees[k].predict(features[j]);
						rv[j] /= trees.S;
					}
				}, i);
			REP(i, config.threadsNo) threads[i].join();
			delete[] threads;
		}
		return rv;
	}
	
	template <class T> VC<T> predictOOB(VC<VC<T>> &features) {
		assert(config.saveChosenSamples);
		assert(trees.S);
		assert(features.S == trees[0].samplesChosen.S);
		
		VC<T> rv(features.S);
		VI no(features.S);
		
		for (auto tree : trees) REP(i, tree.samplesChosen.S) if (tree.samplesChosen[i] == 0) {
			rv[i] += tree.predict(features[i]);
			no[i]++;
		}
		
		REP(i, features.S) rv[i] /= max(1, no[i]);
		return rv;
	}
	
};

class BoostedForest : public TreeEnsemble {
	public:

	VD currentResults;

	void clear() {
		trees.clear();
		trees.shrink_to_fit();
		currentResults.clear();
	}
	
	template <class T> void train(VC<VC<T>> &features, VC<T> &results, MLConfig &_config, int treesMultiplier = 1) {
		double startTime = getTime();
		config = _config;
		
		int treesNo = treesMultiplier * config.treesNo;
		
		if (currentResults.S == 0) currentResults = VD(results.S);
		
		if (config.threadsNo == 1) {
			VC<T> residuals(config.lossFunction == MLConfig::LOG ? 2 * results.S : results.S);
			REP(i, treesNo) {	
				if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
				if (config.lossFunction == MLConfig::LOG) {
					REP(j, results.S) {
						double pred = 1 / (1 + exp(-currentResults[j]));
						residuals[j] = pred - results[j];
						residuals[features.S + j] = max(1e-12, pred * (1 - pred));
					}
				} else {
					REP(j, results.S) residuals[j] = results[j] - currentResults[j];
				}
				trees.PB(createTree(features, residuals, config, gRNG.next()));
				REP(j, results.S) currentResults[j] += trees[trees.S-1].predict(features[j]) * config.shrinkage;
			}
		} else {
			//TODO: improve MT speed
			mutex mutex;
			for (int i = 0; i < treesNo; i += config.threadsNo) {
				if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
				
				int usedThreads = min(config.threadsNo, treesNo - i);
				VC<T> residuals(config.lossFunction == MLConfig::LOG ? 2 * results.S : results.S);
				if (config.lossFunction == MLConfig::LOG) {
					REP(j, results.S) {
						double pred = 1 / (1 + exp(-currentResults[j]));
						residuals[j] = pred - results[j];
						residuals[features.S + j] = max(1e-12, pred * (1 - pred));
					}
				} else {
					REP(j, results.S) residuals[j] = results[j] - currentResults[j];
				}
				
				thread *threads = new thread[config.threadsNo];
				REP(j, usedThreads) 
					threads[j] = thread([&] {
						mutex.lock();
						int seed = gRNG.next();
						mutex.unlock();
		
						auto tree = createTree(features, residuals, config, seed);
						VD pred(results.S);
						REP(k, pred.S) pred[k] = tree.predict(features[k]) * config.shrinkage;
						
						mutex.lock();
						trees.PB(tree);
						REP(k, pred.S) currentResults[k] += pred[k];
						mutex.unlock();
					});
					
				REP(j, usedThreads) threads[j].join();
				delete[] threads;
			}
		}
		
		if (config.computeImportances) {
			importances = VD(features[0].S);
			for (DecisionTree tree : trees)
				REP(i, importances.S)
					importances[i] += tree.importances[i];
			double sum = 0;
			REP(i, importances.S) sum += importances[i];
			REP(i, importances.S) importances[i] /= sum;
		}
	}
	
	template <class T>
	double predict(VC<T> &features, int treeLo = 0, int treeHi = -1) {
		assert(trees.S);
		if (treeHi == -1) treeHi = trees.S;
	
		double sum = 0;
		if (config.threadsNo == 1) {
			FOR(i, treeLo, treeHi) sum += trees[i].predict(features);
		} else {
			thread *threads = new thread[config.threadsNo];
			VD sums(config.threadsNo);
			int order = 0;
			REP(i, config.threadsNo) 
				threads[i] = thread([&](int offset) {
					for (int j = treeLo + offset; j < treeHi; j += config.threadsNo) 
						sums[offset] += trees[j].predict(features);
				}, i);
			REP(i, config.threadsNo) threads[i].join();
			REP(i, config.threadsNo) sum += sums[i];
			delete[] threads;
		}
		sum *= config.shrinkage;
		if (config.lossFunction == MLConfig::LOG) sum = 1 / (1 + exp(-sum));
		return sum;
	}
	
	template <class T>
	VD predict(VC<VC<T>> &features, int treeLo = 0, int treeHi = -1) {
		VD rv(features.S);
		REP(i, features.S) rv[i] = predict(features[i], treeLo, treeHi);
		return rv;
	}
};

class BoostedForestEnsemble {
	VC<BoostedForest> vBF;
	VVD testResults;
	
public:
		
	template <class T> void train(VC<VC<T>> &features, VC<T> &results, MLConfig &config, int folds = 5) {
		double startTime = getTime();
		
		RNG r(features.S);
		
		vBF = VC<BoostedForest>(folds);
		VI order(features.S);
		REP(i, order.S) order[i] = i;
		REP(i, order.S) swap(order[i], order[i + r.next(order.S - i)]);
		
		int stepSize = config.treesNo;
		
		VC<VC<VC<T>>> trainFeatures(folds);
		VC<VC<T>> trainResults(folds);
		VC<VC<VC<T>>> testFeatures(folds);
		VC<VC<T>> testTruth(folds);
		testResults = VVD(folds);
		REP(i, folds) REP(j, features.S) {
			if (j >= i * features.S / folds && j < (i + 1) * features.S / folds) {
				testFeatures[i].PB(features[j]);
				testTruth[i].PB(results[j]);
				testResults[i].PB(0);
			} else {
				trainFeatures[i].PB(features[j]);
				trainResults[i].PB(results[j]);
			}
		}
		
		VD errors;
		int bestStep = -1;
		int step = -1;
		while (true) {
			step++;
			double error = 0;
			REP(i, folds) {
				vBF[i].train(trainFeatures[i], trainResults[i], config);
				REP(j, testFeatures[i].S) {
					testResults[i][j] += vBF[i].predict(testFeatures[i][j], vBF[i].trees.S - stepSize, -1);
					T diff = testTruth[i][j] - testResults[i][j];
					error += diff * diff;
				}
			}
			errors.PB(error);
			if (bestStep == -1 || errors[bestStep] > error)
				bestStep = step;
			if (step > bestStep + 15) break;
			if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
		}
		
		int removeNo = vBF[0].trees.S - (bestStep + 1) * stepSize;
		REP(i, folds) REP(j, removeNo) vBF[i].trees.pop_back();
		
		DB(vBF[0].trees.S);
	}
	
	template <class T> double predict(VC<T> &features) {
		double rv = 0;
		REP(i, vBF.S) rv += vBF[i].predict(features);
		return rv / vBF.S;
	}
	
	template <class T> VD predict(VC<VC<T>> &features) {
		VD rv(features.S);
		REP(i, features.S) rv[i] = predict(features[i]);
		return rv;
	}
	
	VD predictOOB() {
		VD rv;
		for (auto v : testResults) for (double d : v) rv.PB(d);
		return rv;
	}
};

template <class T> VC<VC<T>> multiply(VC<VC<T>> &A, VC<VC<T>> &B) {
	assert(A[0].S == B.S);
	VC<VC<T>> C(A.S, VC<T>(B[0].S, 0));
	REP(k, A[0].S) REP(i, A.S) REP(j, B[0].S) C[i][j] += A[i][k] * B[k][j];
	return C;
}

template <class T> VC<VC<T>> transpose(VC<VC<T>> &A) {
	VC<VC<T>> X(A[0].S, VC<T>(A.S, 0));
	REP(i, A.S) REP(j, A[0].S) X[j][i] = A[i][j];
	return X;
}

template <class T> VC<VC<T>> matrix(VC<T> &v) {
	VC<VC<T>> X(v.S, VC<T>(1));
	REP(i, v.S) X[i][0] = v[i];
	return X;
}

template <class T> VC<VC<T>> solveLeastSquares(VC<VC<T>> &A, VC<VC<T>> &B, double alpha = 0) {
	assert(A.S == B.S);

	VC<VC<T>> AT = transpose(A);
	VC<VC<T>> M0 = multiply(AT, A);
	int n = M0.S;
	REP(i, n) M0[i][i] += alpha;
	
	VC<VC<T>> L(n, VC<T>(n, 0));
	bool valid = true;
	
	REP(j, n) {
		double d = 0.0;
		REP(k, j) {
			double s = 0.0;
			REP(i, k) s += L[k][i] * L[j][i];
			s = (M0[j][k] - s) / L[k][k];
			L[j][k] = s;
			d += s * s;
		}
		d = M0[j][j] - d;
		// if (d <= 0) DB(d);
		if (d == 0) d = 1e-7;
		valid &= d > 0;
		L[j][j] = sqrt(max(d, 0.0));
		FOR(k, j + 1, n) L[j][k] = 0.0;
	}
	
	if (!valid) return VC<VC<T>>();
	
	VC<VC<T>> M1 = multiply(AT, B);
	int nx = B[0].S;

	REP(k, n) REP(j, nx) {
		REP(i, k) M1[k][j] -= M1[i][j] * L[k][i];
		M1[k][j] /= L[k][k];
	}
	
	for (int k = n - 1; k >= 0; k--) REP(j, nx) {
		FOR(i, k + 1, n) M1[k][j] -= M1[i][j] * L[i][k];
		M1[k][j] /= L[k][k];
	}
      
	return M1;
}

template <class T> VC<T> solveLeastSquares(VC<VC<T>> &A, VC<T> &b, double alpha = 0) {
	VC<VC<T>> B(b.S, VC<T>(1));
	REP(i, B.S) B[i][0] = b[i];
	VC<VC<T>> X = solveLeastSquares(A, B, alpha);
	if (X.S == 0) return VC<T>();
	VC<T> R(X.S);
	REP(i, R.S) R[i] = X[i][0];
	return R;
}

class LinearRegression {
public:
	VD weights;
	
	template <class T> void train(VC<VC<T>> &features, VC<T> &results, double regularization = 0) {
		assert(features.S == results.S);
		weights = solveLeastSquares(features, results, regularization);
	}
	
	template <class T> double predict(VC<T> &features) {
		T rv = 0;
		REP(i, features.S) rv += features[i] * weights[i];
		return rv;
	}
	
};



struct Utils {

	static VI generateSequence(int n, int start = 0, int step = 1) {
		VI rv(n);
		REP(i, n) rv[i] = start + i * step;
		return rv;
	}
	
	static VI generatePermutation(int n, int seed) {
		VI rv = generateSequence(n);
		RNG rng(seed);
		REP(i, n) swap(rv[i], rv[i + rng.next(n - i)]);
		// REP(i, n) swap(rv[i], rv[i + Utils::rng.next(n - i)]);
		return rv;
	}

	static VI generateSubset(int n, int k, int seed) {
		VI v = generatePermutation(n, seed);
		VI rv(k);
		REP(i, k) rv[i] = v[i];
		return rv;
	}
	
	template <class T>
	static pair<VC<T>, VC<T>> splitData(VC<T> &data, double split, int seed) {
		VI p = generatePermutation(data.S, seed);
		VC<T> trainData;
		VC<T> testData;
		int trainSamples = (int)(data.S * split);
		REP(i, trainSamples) trainData.PB(data[p[i]]);
		FOR(i, trainSamples, data.S) testData.PB(data[p[i]]);
		return MP(trainData, testData);
	}
	
	template <class T>
	static VC<pair<VC<T>, VC<T>>> generateKFold(VC<T> &data, int folds) {
		VC<pair<VC<T>, VC<T>>> rv;
		REP(fold, folds) {
			VC<T> train;
			VC<T> test;
			REP(i, data.S) {
				if (i >= fold * data.S / folds && i < (fold + 1) * data.S / folds) {
					test.PB(data[i]);
				} else {
					train.PB(data[i]);
				}
			}
			rv.PB(MP(train, test));
		}
		return rv;
	}
	
	template <class T>
	static double calcMSE(VC<T> &v0, VC<T> &v1) {
		assert(v0.S == v1.S);
		double rv = 0;
		REP(i, v0.S) rv += (v0[i] - v1[i]) * (v0[i] - v1[i]);
		return rv / v0.S;
	}
	
	template <class T>
	static double calcMin(VC<T> &v) {
		return v.S == 0 ? 0 : *min_element(ALL(v));
	}
	
	template <class T>
	static double calcMax(VC<T> &v) {
		return v.S == 0 ? 0 : *max_element(ALL(v));
	}
	
	template <class T>
	static double calcMean(VC<T> &v) {
		if (v.S == 0) return 0;
		
		double rv = 0;
		REP(i, v.S) rv += v[i];
		return rv / v.S;
	}
	
	template <class T>
	static double calcSTD(VC<T> &v) {
		if (v.S == 0) return 0;
		
		double mean = calcMean(v);
		
		double rv = 0;
		REP(i, v.S) rv += (v[i] - mean) * (v[i] - mean);
		return sqrt(rv / v.S);
	}
	
	template <class T>
	static double calcCorrelation(VC<T> &a, VC<T> &b) {
		assert(a.S == b.S);
		double meana = calcMean(a);
		double meanb = calcMean(b);
		double stda = calcSTD(a);
		double stdb = calcSTD(b);
		if (stda == 0 || stdb == 0) return 0;
		double sum = 0;
		REP(i, a.S)	sum += (a[i] - meana) * (b[i] - meanb);
		double rv = sum / a.S / stda / stdb;
		return rv;
	}
	
	template <class T>
	static double calcAUC(VC<T> pred, VC<T> truth) {
		assert(pred.S == truth.S);
		double sum = 0;
		double total = 0;
		REP(i, pred.S) if (truth[i] == 1) REP(j, pred.S) if (truth[j] == 0) {
			total++;
			if (pred[i] == pred[j]) sum += 0.5;
			if (pred[i] >  pred[j]) sum += 1;
		}
		return sum / total;
	}
	
	template <class T>
	static double calcFastAUC(VC<T> pred, VC<T> truth) {
		assert(pred.S == truth.S);
		VC<pair<double,double>> vp;
		REP(i, pred.S) vp.PB(MP(pred[i], truth[i]));
		sort(vp.rbegin(), vp.rend());
		int bad = 0;
		REP(i, truth.S) bad += truth[i] == 0;
		int good = pred.S - bad;
		
		double sum = 0;
		double total = bad * good;
		REP(i, vp.S) {
			if (vp[i].Y == 1) {
				sum += bad;
			} else {
				bad--;
			}
		}
		
		return sum / total;
	}
	
	template <class T>
	static VD toRank(VC<T> v) {
		VC<pair<T,int>> vp(v.S);
		REP(i, vp.S) vp[i] = MP(v[i], i);
		sort(ALL(vp));
		VD rv(v.S);
		REP(i, v.S) rv[vp[i].Y] = i;
		return rv;
	}
	
	template <class T>
	static VC<VC<T>> selectColumns(VC<VC<T>> &data, VI columns) {
		VC<VC<T>> rv(data.S, VC<T>(columns.S));
		REP(i, data.S) REP(j, columns.S) rv[i][j] = data[i][columns[j]];
		return rv;
	}
};


VVD lossMatrix = {{0.00, 0.20, 0.70},
                  {0.50, 0.00, 0.01},
				  {1.00, 0.01, 0.00}};
				  
VS classNames = {"No", "Maybe", "Yes"};

int findBestClass(VD &p) {
	double bv = 1e9;
	int best = -1;
	REP(i, 3) {
		double av = 0;
		REP(j, 3) av += p[j] * lossMatrix[i][j];
		if (av < bv) {
			bv = av;
			best = i;
		}
	}
	return best;
}

int classToID(string &s) {
	REP(i, 3) if (s == classNames[i]) return i;
	assert(false);
	return -1;
}



struct Sample {
	int productID;
	int customerID;
	int date;
	double price;
	double sales;
	//int region;
	//int warehouse;
	int customerZip;
	int customerSegment;
	int customerType1;
	int customerType2;
	int customerAccount;
	int customerFirstOrderDate;
	int productClass1;
	int productClass2;
	int productClass3;
	int productClass4;
	int brand;
	int attribute;
	int salesUnit;
	double weight;
	double boxesSold;
	double cost;
	int measure;
	int source;
	int priceMethod;
	int truth = -1;
	
	double amount;
	
	int convertDate(string &s) {
		int y, m, d;
		sscanf(s.c_str(), "%d-%d-%d", &y, &m, &d);
		int x = d + m * 31 + y * 31 * 12; 
		return x;
	}
	
	void print() {
		DB(productID);
		DB(customerID);
		DB(date);
		DB(price);
		DB(sales);
		DB(customerZip);
		DB(customerSegment);
		DB(customerType1);
		DB(customerType2);
		DB(customerAccount);
		DB(customerFirstOrderDate);
		DB(productClass1);
		DB(productClass2);
		DB(productClass3);
		DB(productClass4);
		DB(brand);
		DB(attribute);
		DB(salesUnit);
		DB(weight);
		DB(boxesSold);
		DB(cost);
		DB(measure);
		DB(source);
		DB(priceMethod);
		DB(amount);
		DB(truth);
	}
	
	Sample() { }
	
	Sample(string &s) {
		VS v = splt(s, ',');
		assert(v.S == 28 || v.S == 29);
		productID = atoi(v[0].c_str());
		customerID = atoi(v[1].c_str());
		date = convertDate(v[2]);
		price = atof(v[3].c_str());
		sales = atof(v[4].c_str());
		customerZip = atoi(v[7].c_str());
		customerSegment = v[8][0] - 'A';
		customerType1 = atoi(v[10].c_str());
		customerType2 = v[11][0] - 'A';
		customerAccount = v[13][0] - 'A';
		customerFirstOrderDate = convertDate(v[14]);
		productClass1 = atoi(v[15].c_str());
		productClass2 = atoi(v[16].c_str());
		productClass3 = atoi(v[17].c_str());
		productClass4 = atoi(v[18].c_str());
		brand = v[19] == "IN_HOUSE";
		attribute = atoi(v[20].c_str());
		salesUnit = v[21] == "Y";
		weight = atof(v[22].c_str());
		boxesSold = atof(v[23].c_str());
		cost = atof(v[24].c_str());
		measure = v[25] == "B" ? 0 : v[25] == "EA" ? 1 : 2;
		source =  v[26][0] - 'A';
		priceMethod = atoi(v[27].c_str());
		truth = v.S == 28 ? -1 : classToID(v[28]);
		
		amount = salesUnit ? weight : boxesSold;
	}
	
};

VI getDataIDs(VS &data) {
	VI rv(data.S);
	map<PII,int> idmap;
	REP(i, data.S) {
		Sample smp(data[i]);
		PII p = MP(smp.productID, smp.customerSegment);
		if (idmap.count(p) == 0) {
			int no = idmap.S;
			idmap[p] = no;
		}
		rv[i] = idmap[p];
	}
	return rv;
}

VC<VC<Sample>> parseData(VS &data) {
	VC<VC<Sample>> rv;
	map<PII,int> idmap;
	REP(i, data.S) {
		Sample smp(data[i]);
		PII p = MP(smp.productID, smp.customerSegment);
		if (idmap.count(p) == 0) {
			int no = idmap.S;
			idmap[p] = no;
			rv.PB(VC<Sample>());
		}
		rv[idmap[p]].PB(smp);
	}
	return rv;
}

VC<VC<Sample>> parseGroup(VC<Sample> &group) {
	VC<VC<Sample>> rv;
	map<int,int> idmap;
	for (Sample &s : group) {
		if (idmap.count(s.customerID) == 0) {
			int no = idmap.S;
			idmap[s.customerID] = no;
			rv.PB(VC<Sample>());
		}
		rv[idmap[s.customerID]].PB(s);
	}
	return rv;
}

RNG rng;

struct Customer {
	VI truthCount = VI(3);
	int total;
	
	void clear() {
		REP(i, 3) truthCount[i] = 0;
		total = 0;
	}
};

Customer allCustomers[200];

VD extractFeaturesCustomer(VC<Sample> &group) {
	VD rv;
	
	VD amount;
	VD salesRatio;
	VD costRatio;
	VD date;
	VD price;
	VD markupRatio;
	for (Sample &s : group) {
		if (s.amount == 0) continue;
		costRatio.PB(s.cost / s.amount);
		salesRatio.PB(s.sales / s.amount);
		amount.PB(s.amount);
		date.PB(s.date);
		price.PB(s.price);
		markupRatio.PB((s.sales - s.cost) / s.sales);
	}
	sort(ALL(date));
	VD dateScore;
	FOR(i, 1, date.S) {
		double expDate = date[0] + (date.back() - date[0]) * i / (date.S - 1);
		// dateScore.PB((expDate - date[i])/(date.back()-date[0]));
		dateScore.PB((expDate - date[i]) * (expDate - date[i]));
	}
	rv.PB(Utils::calcMax(salesRatio) / Utils::calcMin(salesRatio));
	rv.PB(Utils::calcMax(costRatio) / Utils::calcMin(costRatio));
	rv.PB(Utils::calcCorrelation(salesRatio, costRatio));
	rv.PB(Utils::calcCorrelation(price, costRatio));
	rv.PB(Utils::calcMean(dateScore));
	rv.PB(Utils::calcSTD(dateScore));
	rv.PB(Utils::calcMean(amount));
	rv.PB(Utils::calcSTD(amount));
	rv.PB(Utils::calcMean(price));
	rv.PB(Utils::calcSTD(price));
	rv.PB(Utils::calcMin(markupRatio));
	rv.PB(Utils::calcMax(markupRatio));
	rv.PB(Utils::calcMean(markupRatio));
	rv.PB(Utils::calcSTD(markupRatio));
	rv.PB(group.S);
	rv.PB(group[0].customerFirstOrderDate);
	return rv;
}

double safediv(double a, double b) {
	return b == 0 ? 0 : a / b;
}

BoostedForest CBF[3];
VD extractFeatures(VC<Sample> &group, vector<int> dropFeatures, double skipSamples = 0.0, double skipCustomers = 0.0) {
	VD rv;
	
	VD sales;
	VD prices;
	VD amount;
	VD cost;
	VD date;
	VD costRatio;
	VD salesRatio;
	VD markup;
	VD markupAvg;
	VD markupRatio;
	VD priceMethod;
	VD source;
	VD customerAccount;
	VD customerType1;
	VD customerType2;
	VD boxesSold;
	VD weight;
	VPDD dateCostRatio;
	VPDD dateSalesRatio;
	VVD customerTruths(3);
	int empty = 0;
	
	for (Sample &s : group) {
		if (rng.nextDouble() < skipSamples) continue;
		if (s.amount == 0) {
			empty++;
			continue;
		}
		sales.PB(s.sales);
		prices.PB(s.price);
		amount.PB(s.amount);
		cost.PB(s.cost);
		costRatio.PB(safediv(s.cost, s.amount));
		salesRatio.PB(safediv(s.sales, s.amount));
		date.PB(s.date);
		markup.PB(s.sales - s.cost);
		markupAvg.PB(safediv(s.sales - s.cost, s.amount));
		markupRatio.PB(safediv(s.sales - s.cost, s.sales));
		priceMethod.PB(s.priceMethod);
		source.PB(s.source);
		customerAccount.PB(s.customerAccount);
		customerType1.PB(s.customerType1);
		customerType2.PB(s.customerType2);
		boxesSold.PB(s.boxesSold);
		weight.PB(s.weight);
		dateCostRatio.PB(MP(s.date, safediv(s.cost, s.amount)));
		dateSalesRatio.PB(MP(s.date, safediv(s.sales, s.amount)));
		REP(i, 3) customerTruths[i].PB(1.0 * allCustomers[s.customerID].truthCount[i] / allCustomers[s.customerID].total);
	}
	if (group.S && sales.S == 0) return extractFeatures(group, dropFeatures, max(skipSamples - 0.1, 0.0), max(skipCustomers - 0.1, 0.0));
	
	double costChanges = 0;
	
	int uniqueCosts = 1;
	sort(ALL(dateCostRatio));
	FOR(i, 1, dateCostRatio.S) {
		if (abs(dateCostRatio[i].Y - dateCostRatio[i-1].Y) > 0) uniqueCosts++;
		costChanges += abs(dateCostRatio[i].Y - dateCostRatio[i-1].Y);
	}

	int uniqueSales = 1;
	sort(ALL(dateSalesRatio));
	FOR(i, 1, dateSalesRatio.S) if (abs(dateSalesRatio[i].Y - dateSalesRatio[i-1].Y) > 0) uniqueSales++;

	assert(group.S > 0);
	
	
	rv.PB(costChanges / Utils::calcMean(costRatio));
	rv.PB(group.S);
	rv.PB(group[0].customerSegment);
	rv.PB(group[0].productID);
	rv.PB(safediv(uniqueCosts, cost.S));
	rv.PB(safediv(uniqueSales, sales.S));
	rv.PB(safediv(Utils::calcMax(salesRatio), Utils::calcMin(salesRatio)));
	rv.PB(safediv(Utils::calcMax(costRatio), Utils::calcMin(costRatio)));
	rv.PB(Utils::calcCorrelation(salesRatio, costRatio));
	rv.PB(safediv(empty, group.S));
	rv.PB(Utils::calcMean(sales));
	rv.PB(Utils::calcMean(prices));
	rv.PB(Utils::calcMean(amount));
	rv.PB(Utils::calcMean(cost));
	rv.PB(Utils::calcMean(markup));
	rv.PB(Utils::calcMean(markupAvg));
	rv.PB(Utils::calcMean(markupRatio));
	rv.PB(Utils::calcMean(priceMethod));
	rv.PB(Utils::calcSTD(priceMethod));
	rv.PB(Utils::calcMean(source));
	rv.PB(Utils::calcMean(customerAccount));
	rv.PB(Utils::calcMean(customerType1));
	rv.PB(Utils::calcMean(customerType2));
	rv.PB(Utils::calcSTD(sales));
	rv.PB(Utils::calcSTD(prices));
	rv.PB(Utils::calcSTD(amount));
	rv.PB(Utils::calcSTD(cost));
	rv.PB(Utils::calcSTD(markup));
	rv.PB(Utils::calcSTD(markupAvg));
	rv.PB(Utils::calcSTD(markupRatio));
	rv.PB(Utils::calcSTD(source));
	rv.PB(Utils::calcSTD(customerAccount));
	rv.PB(Utils::calcSTD(customerType1));
	rv.PB(Utils::calcSTD(customerType2));
	rv.PB(group[0].attribute);
	rv.PB(group[0].brand);
	rv.PB(group[0].productClass1);
	rv.PB(group[0].productClass2);
	rv.PB(group[0].productClass3);
	rv.PB(group[0].productClass4);
	
	sort(dropFeatures.begin(), dropFeatures.end());
	
	for(int i = dropFeatures.size() - 1; i >= 0; i--)
	{
	    rv.erase(rv.begin() + dropFeatures[i]);
	}
	
	//for(int i= 0; i < dropFeatures.size(); i++)
	//    rv[dropFeatures[i]] = 0;
	
	VC<VC<Sample>> customers = parseGroup(group);
	// rv.PB(customers.S);
	
	if (!USE_BASIC_FEATURES) rv.clear();
	
	if (USE_CUSTOMER_PREDICTIONS) {
		VVD customersPred(3);
		for (auto &customer : customers) {
			if (rng.nextDouble() < skipCustomers) continue;
			VD f = extractFeaturesCustomer(customer);
			REP(i, 3) customersPred[i].PB(CBF[i].predict(f));
		}
		REP(i, 3) {
			// rv.PB(Utils::calcMin(customersPred[i]));
			// rv.PB(Utils::calcMax(customersPred[i]));
			rv.PB(Utils::calcMean(customersPred[i]));
			rv.PB(Utils::calcSTD(customersPred[i]));
		}
	}
	
	return rv;
}

VD modelBF1(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.shrinkage = 0.01;
	cfg.treesNo = 600;
	cfg.randomFeatures = {6};
	cfg.randomPositions = {1};
	cfg.maxLevel = 1;
	cfg.lossFunction = MLConfig::LOG;
	
	BoostedForest BF;
	BF.train(trainData, truth, cfg);
	return BF.predict(testData);
}

VD modelBF2(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.shrinkage = 0.01;
	cfg.treesNo = 600;
	cfg.randomFeatures = {16};
	cfg.randomPositions = {1};
	cfg.maxLevel = 1;
	cfg.lossFunction = MLConfig::LOG;
	
	BoostedForest BF;
	BF.train(trainData, truth, cfg);
	return BF.predict(testData);
}

VD modelBF3(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.shrinkage = 0.001;
	cfg.treesNo = 6000;
	cfg.randomFeatures = {10};
	cfg.randomPositions = {1};
	cfg.maxLevel = 1;
	cfg.lossFunction = MLConfig::LOG;
	
	BoostedForest BF;
	BF.train(trainData, truth, cfg);
	return BF.predict(testData);
}

VD modelRF1(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.treesNo = 200;
	cfg.randomFeatures = {6};
	cfg.randomPositions = {1};
	cfg.lossFunction = MLConfig::MQE;
	
	RandomForest RF;
	RF.train(trainData, truth, cfg);
	return RF.predict(testData);
}

VD modelRF2(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.treesNo = 500;
	cfg.randomFeatures = {10};
	cfg.randomPositions = {1};
	cfg.lossFunction = MLConfig::MSE;
	
	RandomForest RF;
	RF.train(trainData, truth, cfg);
	return RF.predict(testData);
}

VD modelRF3(VVD &trainData, VD &truth, VVD &testData) {
	MLConfig cfg;
	cfg.threadsNo = THREADS_NO;
	cfg.treesNo = 500;
	cfg.randomFeatures = {10};
	cfg.randomPositions = {1};
	cfg.lossFunction = MLConfig::MQE;
	
	RandomForest RF;
	RF.train(trainData, truth, cfg);
	return RF.predict(testData);
}

class ElectronicPartsClassification {public: 
VS classifyParts(VS &trainingData, VS &testingData, VC<VC<Sample>> &trainGroups, VC<VC<Sample>> &testGroups, vector<int> dropFeatures) {
	VVD trainData;
	VVD trainTruth(3);
	VVD testData;
	VVD testPred(3);
	
	REP(i, 200) allCustomers[i].clear();
	
	for (auto &v : trainGroups) {
		for (Sample &s : v) {
			allCustomers[s.customerID].total++;
			allCustomers[s.customerID].truthCount[s.truth]++;
		}
	}
		
		
	if (USE_CUSTOMER_PREDICTIONS) {
		VVD trainDataCustomers;
		VVD trainTruthCustomers(3);
		for (auto group : trainGroups) {
			auto customers = parseGroup(group);
			for (auto customer : customers) {
				trainDataCustomers.PB(extractFeaturesCustomer(customer));
				REP(i, 3) trainTruthCustomers[i].PB(customer[0].truth == i);
			}
		}
	
		MLConfig cfg;
		cfg.threadsNo = THREADS_NO;
		cfg.shrinkage = 0.02;
		cfg.treesNo = 200;
		cfg.randomFeatures = {5};
		cfg.randomPositions = {1};
		cfg.maxLevel = 2;
		cfg.lossFunction = MLConfig::LOG;
		
		REP(i, 3) CBF[i].clear();
		REP(i, 3) CBF[i].train(trainDataCustomers, trainTruthCustomers[i], cfg);
	}
	
	for (auto &v : trainGroups) {
		REP(step, TRAIN_AUGMENTS) {
			double r1 = rng.nextDouble() * 0.95;
			double r2 = rng.nextDouble() * 0.95;
			if (step == 0) r1 = r2 = 0;
			trainData.PB(extractFeatures(v, dropFeatures, r1, r2));
			REP(i, 3) trainTruth[i].PB(v[0].truth == i);
		}
	}
	
	set<int> productIDs;
	map<int,VI> productGroups[2];
	
	REP(i, testGroups.S) {
		auto &v = testGroups[i];
		if (productIDs.count(v[0].productID) == 0) {
			productIDs.insert(v[0].productID);
			productGroups[0][v[0].productID] = VI();
			productGroups[1][v[0].productID] = VI();
		}
		REP(step, TEST_AUGMENTS) {
			productGroups[v[0].customerSegment][v[0].productID].PB(i * TEST_AUGMENTS + step);
			double r1 = rng.nextDouble() * 0.95;
			double r2 = rng.nextDouble() * 0.95;
			if (step == 0) r1 = r2 = 0;
			testData.PB(extractFeatures(v, dropFeatures, r1, r2));
		}
	}
	
	REP(i, 3) {
		VVD results = {
			// modelBF1(trainData, trainTruth[i], testData),
			// modelBF2(trainData, trainTruth[i], testData),
			modelBF3(trainData, trainTruth[i], testData),
			// modelRF1(trainData, trainTruth[i], testData),
			modelRF2(trainData, trainTruth[i], testData),
			modelRF3(trainData, trainTruth[i], testData),
		};
		
		REP(k, results[0].S) {
			double sum = 0;
			REP(j, results.S) sum += results[j][k];
			testPred[i].PB(sum / results.S);
		}
	}
	
	VS rv;
	for (int id : productIDs) {
		string s[2];
    double PV[6];

		REP(i, 2) {
			if (productGroups[i][id].S == 0) {
				s[i] = "NA";
				continue;
			}
			VD p(3, 0.0);
			for (int x : productGroups[i][id]) {
				REP(j, 3) p[j] += testPred[j][x];
			}
			REP(j, 3) p[j] /= productGroups[i][id].S;

      for(int j = 0; j < 3; j++)
        PV[i * 3 + j] = p[j];

			s[i] = classNames[findBestClass(p)];
		}
		rv.PB(i2s(id) + "," + s[0] + "," + s[1] + "," + std::to_string(PV[0]) + "," + std::to_string(PV[1]) + "," + std::to_string(PV[2]) + "," + std::to_string(PV[3]) + "," + std::to_string(PV[4]) + "," + std::to_string(PV[5]));
	}
	return rv;
}

VS classifyParts(VS &trainingData, VS &testingData, vector<int> dropFeatures) {
	VC<VC<Sample>> trainGroups = parseData(trainingData);
	VC<VC<Sample>> testGroups = parseData(testingData);
	return classifyParts(trainingData, testingData, trainGroups, testGroups, dropFeatures);
}

};


import numpy as np
#Author: hf-z

class GeneticFeatureSelection(object):
    '''A simple and easy-to-use genetic algorithm.

    Genetic Algorithm.

    Attributes:
        F: fit function, use to calculate the fitness value of chromosome.
        bits_num: the length of each chromosome.
        N: the number of chromosomes.
        pc: the matting ratio.
        pm: the mutation rate.
        max_iter: maximum iterations.
        generate_function: use which function to generate initial chromosomes, RND_INT or RND_STR.
    '''

    def __init__(self, F, bits_num, N=6, pc=1, pm=0, max_iter=1000, generate_function='RND_STR'):
        self.F = F
        self.bits_num = bits_num
        self.N = N
        self.pc = pc
        self.pm = pm
        self.max_iter = max_iter
        self.generate_function = generate_function

    def _generate_population_based_rndint(self):
        max_value = 2 ** self.bits_num
        population = np.random.choice(np.arange(max_value), self.N, replace=False)
        bin_population = [bin(x) for x in population]
        final_population = []
        for x in bin_population:
            x = x[2:]
            while (len(x) < self.bits_num):
                x = '0' + x
            final_population.append(x)
        final_population = np.array(final_population)
        return final_population

    def _generate_population_based_rndstr(self):
        population = []
        for i in range(self.N):
            s = np.full(self.bits_num, '0')
            rand_inds = np.random.choice(np.arange(self.bits_num), np.random.randint(1, self.bits_num), replace=False)
            s[rand_inds] = '1'
            population.append(''.join(s))
        return np.array(population)

    def genetic(self,*kw):
        '''Evolution Procedure

        :param kw: the same as user-defined fitness function F(chromosome, *kw)
        :return:
            res: a tuple (final solution, the fitness value of final solution).
            max_fit_values: the maximum fitness value of each iteration.
            avg_fit_values: the average fitness value of each iteration.
        '''
        if self.generate_function=='RND_INT':
            population=self._generate_population_based_rndint()
        elif self.generate_function=='RND_STR':
            population = self._generate_population_based_rndstr()
        else:
            raise NameError(self.generate_function+' is not defined.')
        print(population.shape)

        max_fitness_item = [0, -np.inf]
        res = (0, -np.inf)
        probabilities = np.arange(1, len(population) + 1, 1) / np.sum(np.arange(1, len(population) + 1, 1))
        avg_fit_values = []
        max_fit_values = []
        for iter_times in range(self.max_iter):
            if iter_times % 100 == 0:
                print(iter_times)
            fit_values = self.F(population,kw)

            current_max_fitness_item = [np.argmax(fit_values), np.max(fit_values)]
            if current_max_fitness_item[1] > max_fitness_item[1]:
                max_fitness_item = current_max_fitness_item
                res = (population[max_fitness_item[0]], max_fitness_item[1])

            max_fit_values.append(res[1])
            avg_fit_values.append(np.average(fit_values))
            fit_values_inds = np.argsort(fit_values)
            choice_inds = np.random.choice(fit_values_inds, size=len(population), replace=True, p=probabilities)

            new_population = []
            for ind in choice_inds:
                new_population.append(population[ind])
            new_population = np.array(new_population)

            # mate
            mate_num = int(self.pc * self.N)
            if mate_num % 2 == 1:
                mate_num += 1
            mate_instances_inds = np.random.choice(np.arange(self.N), mate_num, replace=False)
            mate_instances = new_population[mate_instances_inds]
            ind = 0
            while (ind < mate_num - 1):
                mate_instance1_arr = list(mate_instances[ind])
                mate_instance2_arr = list(mate_instances[ind + 1])
                mate_pos = int(np.random.choice(np.arange(self.bits_num), 1, replace=False))
                t = mate_instance1_arr[mate_pos:]
                mate_instance1_arr[mate_pos:] = mate_instance2_arr[mate_pos:]
                mate_instance2_arr[mate_pos:] = t

                mate_instances[ind] = ''.join(mate_instance1_arr)
                mate_instances[ind + 1] = ''.join(mate_instance2_arr)
                ind += 2

            mated_population = new_population
            mated_population[mate_instances_inds] = mate_instances

            # mutation
            variation_num = int(self.pm * self.N) + 1
            variation_instances_inds = np.random.choice(np.arange(self.N), variation_num, replace=False)
            variation_instances = new_population[variation_instances_inds]
            ind = 0
            while (ind < variation_num):
                variation_instance1_arr = list(variation_instances[ind])
                variation_pos = int(np.random.choice(np.arange(self.bits_num), 1, replace=False))
                if variation_instance1_arr[variation_pos] == '1':
                    variation_instance1_arr[variation_pos] = '0'
                else:
                    variation_instance1_arr[variation_pos] = '1'
                variation_instances[ind] = ''.join(variation_instance1_arr)
                ind += 1

            variation_population = mated_population
            variation_population[variation_instances_inds] = variation_instances
            population = variation_population

        return res, max_fit_values, avg_fit_values


if __name__ == "__main__":
    def F(p, *kw):
        res = []
        for i in p:
            t = int(''.join(i), 2)
            res.append(t ** 2)
        return np.array(res)


    gfs = GeneticFeatureSelection(F, bits_num=5, N=6, pc=0.8, pm=0, max_iter=100)
    res, mfs, afs = gfs.genetic()
    print(res)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.xlabel('iterations')
    plt.ylabel('max fitness value')
    plt.plot(mfs)
    plt.subplot(212)

    plt.xlabel('iterations')
    plt.ylabel('average fitness value')
    plt.plot(afs)
    plt.show()

# Genetic-Algorithm
A simple and easy-to-use genetic algorithm framework. All you need to do is defining your fitness function.

## Example
Eg. Find the maximum value of y=x^2, where x is integer and the range is [0,31].
* define a fitness function.
    
    
    def F(chromosomes,*kw):
    '''
    params:
        chromosomes: string list of shape (num_of_chromosomes, ), eg:
                    ['10001','11010','00011','01010']
                    each item of the list is a chromosome.
    '''
        fitness_values = []
        #calculate square of each chromosome.
        for i in p:
            t = int(''.join(i), 2)
            fitness_values.append(t ** 2)
        return np.array(fitness_values)
  
* only need 5 bits to represent all solutions, so set the length of chromosome 5, and set
 the number of chromosomes 6 randomly.


    ga=GeneticAlgorithm(F, bits_num=5, N=6)
* get the solution.


    res,max_fit_values,avg_fit_values=ga.genetic()
    print(res)
    #print result: ('11111',961)
 
 That's all, Done!
 ## Others
 There is an example in source code.
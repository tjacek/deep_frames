import numpy as np
import scipy.misc as image
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import utils 

def load_data(path,batch_size=25):
    all_files=utils.get_all_files(path)
    all_files=utils.append_path(path,all_files)
    images=map(lambda f:image.imread(f),all_files)
    images=map(lambda img:np.reshape(img,(img.size)),images)
    n_batches=get_number_of_batches(batch_size,len(images))
    def get_batch(i):
        return images[i * batch_size: (i+1) * batch_size]
    batches=map(get_batch,range(n_batches))
    batches = [np.array(batch) for batch in batches]
    return batches

def get_number_of_batches(batch_size,n_images):
    n_batches=n_images / batch_size
    if(n_images % batch_size != 0):
        n_batches+=1
    return n_batches

class AutoEncoder(object):
    def __init__(
        self,theano_rng=None,input=None,n_visible=3000,
        n_hidden=500,W=None,bhid=None,bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.init_rng(thean_rng)
        self.init_hidden_layer(W,bhid)
        self.init_visable_layer(bvis)
        self.init_input(input)
        self.params = [self.W, self.b, self.b_prime]

    def init_rng(self,thean_rng):
        self.numpy_rng = numpy.random.RandomState(123)
        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

    def init_hidden_layer(self,W,bhid):
        if not W:
            n_units=self.n_hidden + self.n_visible
            initial_W = numpy.asarray(
                self.numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / n_units),
                    high=4 * numpy.sqrt(6. / n_units),
                    size=(self.n_visible, self.n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        self.W=W
        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
	self.b = bhid

    def init_visable_layer(self,bvis):
	if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        self.b_prime = bvis
        self.W_prime = self.W.T

    def init_input(self,input):
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)

def test_dA(learning_rate=0.1, training_epochs=15,
            batch_size=20):

    datasets = load_data(dataset)
    n_train_batches=len(datasets)
    index = T.lscalar()   
    x = T.matrix('x')  

    da = AutoEncoder()

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: dataset[index]
        }
    )

    start_time = timeit.default_timer()
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

if __name__ == "__main__":
    in_path="/home/user/df/frames/"
    load_data(in_path)

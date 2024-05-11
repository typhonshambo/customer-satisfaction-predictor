import click
import logging

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option(
    '--mode', 
    '-m',
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help='''Choose the mode to run the deployment.
    (`deploy`) - Deploy the model 
    (`predict`) - Make predictions using the deployed model
    (`deploy_and_predict`) - Deploy the model and make predictions using the deployed model
    By default, it is set to `deploy_and_predict`
    '''
)
@click.option(
    '--min-accuracy',
    default=0.92,
    help='Minimum accuracy required for deployment'
)
def run_deployment(mode: str, min_accuracy: float) -> None:
    if mode == DEPLOY:
        '''
        Deploy pipeline
        '''
        logging.info('Deploying the model')
        
    elif mode == PREDICT:
        '''
        Inference pipleline
        '''
        logging.info('Making predictions')
    else:
        logging.info('Deploying the model and making predictions')
import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'this is a hard-to-guess string'


class DevelopmentConfig(Config):
    ENV = 'development'
    DEBUG = True
    DEVELOPMENT = True
    
    
class ProductionConfig(Config):
    ENV = 'production'
    DEBUG = False
    DEVELOPMENT = False

    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig
}
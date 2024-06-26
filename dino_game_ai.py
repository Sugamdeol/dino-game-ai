from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import cv2
import time
from stable_baselines3 import DQN
import gym

# Selenium WebDriver setup
options = webdriver.ChromeOptions()
options.add_argument("--disable-infobars")
options.add_argument("--mute-audio")
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Load the Dino game
driver.get("chrome://dino")
driver.execute_script("Runner.config.ACCELERATION=0")

# Get the game canvas element
canvas = driver.find_element_by_class_name('runner-canvas')
canvas_width = canvas.size['width']
canvas_height = canvas.size['height']

# Function to capture the game screen
def get_screenshot():
    screenshot = driver.get_screenshot_as_png()
    screenshot = np.frombuffer(screenshot, np.uint8)
    image = cv2.imdecode(screenshot, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[canvas.location['y']:canvas.location['y'] + canvas_height,
                  canvas.location['x']:canvas.location['x'] + canvas_width]
    image = cv2.resize(image, (84, 84))
    return np.array(image, dtype=np.uint8)

# Action methods
def jump():
    driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

def duck():
       driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

   # Environment class for the Dino game
   class DinoEnv(gym.Env):
       def __init__(self):
           super(DinoEnv, self).__init__()
           self.action_space = gym.spaces.Discrete(3)  # 0: Do nothing, 1: Jump, 2: Duck
           self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
           self.reset()
       
       def reset(self):
           driver.execute_script("Runner.instance_.restart()")
           time.sleep(0.5)
           return get_screenshot()
       
       def step(self, action):
           if action == 1:
               jump()
           elif action == 2:
               duck()
           time.sleep(0.1)
           observation = get_screenshot()
           score = driver.execute_script("return Runner.instance_.distanceMeter.getActualDistance(Runner.instance_.distanceRan)")
           done = driver.execute_script("return Runner.instance_.crashed")
           reward = float(score) if not done else -100.0
           return observation, reward, done, {}

   # Instantiate the environment and model
   env = DinoEnv()
   model = DQN('CnnPolicy', env, verbose=1)

   # Train the model
   model.learn(total_timesteps=10000)

   # Save the model
   model.save("dino_model")

   # Load and test the model
   model = DQN.load("dino_model")

   obs = env.reset()
   for _ in range(1000):
       action, _states = model.predict(obs)
       obs, rewards, done, info = env.step(action)
       if done:
           obs = env.reset()

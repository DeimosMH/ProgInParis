from BCI_Processor import BCIProcessor 

class Messenger:
    """
    Handles event state (what happened) and processes them by dispatching 
    to the BCIProcessor.
    """
    def __init__(self):
        # Stores the current state of detected actions/events

        self.actions = {} 
        print('Message created')
        # Step 2: Create a dispatch map using the imported BCIProcessor methods
        self.processor_map = {
            'double_blink': BCIProcessor.doubleBlink,
            'triple_blink': BCIProcessor.tripleBlink,
            'blink_right': BCIProcessor.blinkRight,
            'blink_left': BCIProcessor.blinkLeft,
            'look_right': BCIProcessor.lookRight,
            'look_left': BCIProcessor.lookLeft,
            'look_up': BCIProcessor.lookUp,
            'look_down': BCIProcessor.lookDown,
        }

    def set_action(self, action_name: str):
        """Sets an action to be processed by marking it True."""
        if action_name in self.processor_map:
            self.actions[action_name] = True
            print(f"Messenger: Received event '{action_name}'.")
        else:
            print(f"Messenger: Warning! Unknown action '{action_name}'.")

    def process(self):
        """Executes the functions for all currently set actions and then clears the state."""
        processed_count = 0
        
        # Find actions that were set (state is True)
        active_actions = [action for action, state in self.actions.items() if state]
        
        if not active_actions:
            print("Messenger: No actions to process.")
            return

        print("\n--- Starting Event Processing ---")
        for action_name in active_actions:
            # Execute the corresponding BCIProcessor method
            self.processor_map[action_name]()
            processed_count += 1
            
        # Reset the state for the next cycle
        self.actions.clear() 
        print(f"--- Finished processing {processed_count} action(s) ---\n")

    def sendChartData(self, data :list()):
        BCIProcessor.signalData(data)


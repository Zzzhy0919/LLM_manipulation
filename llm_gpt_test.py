import openai
import time

# openai.api_key = "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" #blueYu
# openai.api_key = "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" #tsinghua
openai.api_base="https://ai98.vip/v1"
openai.api_key = "sk-CcCvez7msME97Kwp5cDb5947E8D7466a94Cf400b21285d0d" #taobao

#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
class Chat:
    def __init__(self):
        self._base_prompt = None
        self._context = None
    
    # 打印对话
    def show_conversation(self,msg_list):
        for msg in msg_list:
            if msg['role'] == 'user':
                print(f"\U0001f47b: {msg['content']}\n")
            else:
                print(f"\U0001f47D: {msg['content']}\n")

    # 提示chatgpt
    def ask(self, prompt):
        instructions = '''
            “here are the instruction-templates:(in "[]")
            [
            "Fold the Trousers in half, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers, {which1} side over {which2} side.",
            "Bend the Trousers in half, from {which1} to {which2}.",
            "Crease the Trousers down the middle, from {which1} to {which2}.",
            "Fold the Trousers in half horizontally, {which1} to {which2}.",
            "Make a fold in the Trousers, starting from the {which1} and ending at the {which2}.",
            "Fold the Trousers in half, aligning the {which1} and {which2} sides.",
            "Fold the Trousers, orientating from the {which1} towards the {which2}.",
            "Fold the Trousers in half, with the {which1} side overlapping the {which2}.",
            "Create a fold in the Trousers, going from {which1} to {which2}."
            "Bring the {which1} side of the Trousers towards the {which2} side and fold them in half.",
            "Fold the waistband of the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers neatly, from the {which1} side to the {which2} side.",

            "Fold the Trousers in half vertically from top to bottom.",
            "Create a fold in the Trousers from the waistband to the hem.",
            "Fold the Trousers along the vertical axis, starting from the top.",
            "Fold the Trousers in half lengthwise, beginning at the waistband.",
            "Fold the Trousers vertically, starting at the waistband.",
            "Fold the Trousers in half, starting from the top edge.",
            "Fold the Trousers by bringing the waistband down to meet the hem.",
            "Fold the Trousers in half vertically, starting at the upper edge.",
            "Fold the Trousers by bringing the waistband down to meet the bottom.",
            "Fold the Trousers in half, starting from the top seam.",
            "Fold the Trousers in half, bringing the top towards the hem.",

            "Bring the {which1} side of the Trousers towards the {which2} side and fold them in half.",
            "Fold the waistband of the Trousers in half, from {which1} to {which2}.",
            "Fold the Trousers neatly, from the {which1} side to the {which2} side.",
            "Fold the Trousers, making a crease from the {which1} to the {which2}.",
            ]
            Then there are some additional notes :
            1.{which1} and {which2} can be random in: location_list=['left' ,'right', 'top' ,'bottom' ,'waistband','hem']
            2.you can replace words in location_list with other words in location_list.
            ”
            '''
        instructions2 = '''
            I will give you some widely instrucion,and you should reply me three sentences following the instruction-templates I've gave you.
            For example,
            users:"fold the Trousers to minimum"
            assistant:
            1."Create a fold in the Trousers from the waistband to the hem"
            2."Bring the right side of the Trousers towards the left side and fold them in half."
            3."Fold the Trousers in half vertically, beginning at the top."
            ""


            '''
        assistant1 = f'Got it. I will complete what you give me next.'
        ret = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's top-level instructions and writes low-level instruction sentences according to the instruction-templates I will give you."},
                {"role": "user", "content": instructions},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": instructions2},
                {"role": "user", "content": prompt}
            ]
            
        )
        answer = ret.choices[0].message.content
        return answer

 
if __name__ == "__main__":
    print("start...")
    cha = Chat()
    while True:
        query = input("please input your instruction:")
        ans = cha.ask(query)
        lines = ans.strip().split("\n")

        # 然后去除每行的数字序号和空格
        my_list = [line.split('. ', 1)[1].strip('"') for line in lines if line]
        
        my_list
        print(my_list)

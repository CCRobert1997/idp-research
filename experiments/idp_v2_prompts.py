"""IDP v2 diverse-content prompt pool, organized by the defect TYPE each is meant
to elicit. 6 types x 12 prompts x 70 seeds (0..69) ~= 5040 images.
prompt_type is the track axis for the E3 per-type AUC-vs-step analysis."""

PROMPTS_V2 = {
    "HAND": [
        "a person waving", "a chef holding a knife", "two people shaking hands",
        "a person counting on fingers", "hands typing on a keyboard", "a person playing guitar",
        "a barista making coffee", "a person holding chopsticks", "a painter holding a brush",
        "close-up of hands holding a phone", "a person giving a thumbs up", "a magician's hands",
    ],
    "FACE": [
        "a close-up portrait of a woman", "a man's face, studio lighting", "a child smiling",
        "an elderly person's face", "two people talking face to face", "a person wearing glasses",
        "a portrait, side profile", "a crowd of faces", "a person laughing",
        "a face reflected in a mirror", "twins standing together", "a person with detailed eyes",
    ],
    "LIMB": [
        "a dancer mid-leap", "a yoga instructor in a pose", "a runner sprinting",
        "a gymnast on a balance beam", "a person doing a cartwheel", "a martial artist kicking",
        "a swimmer diving", "a basketball player jumping", "a person stretching",
        "a couple dancing", "a soccer player kicking a ball", "a person climbing a wall",
    ],
    "COUNT": [
        "three red apples on a table", "five wine glasses in a row", "four cats sitting together",
        "two birds on a branch", "six books stacked", "a hand showing three fingers",
        "three identical coffee mugs", "four candles burning", "two dogs playing",
        "five colorful balloons", "three slices of pizza", "a pair of shoes",
    ],
    "PHYS": [
        "a glass of water tipping over", "a bicycle leaning on a wall", "a chair and a table",
        "a stack of plates", "an octopus holding objects", "a spider on a web",
        "a chandelier hanging from a ceiling", "gears of a clock mechanism", "a ladder against a house",
        "a tangle of cables", "a bouquet of many flowers", "scaffolding on a building",
    ],
    "TEXT": [
        "a storefront with a sign that says OPEN", "a book cover titled ADVENTURE",
        "a street sign reading STOP", "a coffee cup with text HELLO", "a t-shirt with the word LOVE",
        "a neon sign saying BAR", "a poster with the word SALE", "a license plate",
        "a chalkboard menu", "a billboard advertisement", "a newspaper headline", "a label on a bottle",
    ],
}
TYPES = list(PROMPTS_V2.keys())
SEEDS_PER_PROMPT = 70

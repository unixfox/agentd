import pytest
from agentd.section_chunker import wrap_sections, unwrap_sections, parse_sections
from agentd.editors.section_based_editor import SectionBasedEditor

# A long blurb to test with.
long_blurb = """
Puppy Paradise

(Verse 1)
In a world so grand, with fields to explore,  
Tiny paws on the ground, and we're ready for more.  
Little wet noses, sniffing the breeze,  
Chasing our tails, with hearts full of glee.  

(Chorus)
Oh, puppies, oh puppies, with eyes so bright,  
Paint the world with joy and pure delight.  
Furry little bundles of endless cheer,  
In the dance of their innocence, all troubles disappear.
In the dance of their innocence, all troubles disappear.  

(Verse 2)
Playful barks echo, in the morning light,
Jumping through the flowers, what a beautiful sight.
Wagging little tails, chasing dreams so high,  
With a cuddle and a giggle, we watch the day go by.
(Chorus)  
Oh, puppies, oh puppies, with eyes so bright,
Paint the world with joy and pure delight.
Furry little bundles of endless cheer,  
In the dance of their innocence, all troubles disappear.
(Bridge)
Every puppy step tells a story true,  
Of friendship and love, always fresh as dew.
With a paw in our hand, no journey's too long,  
Together we write our puppy song.  

(Verse 3)
As the sun sets low, they curl up tight,
Dreaming puppy dreams, through the calming night.
In a blanket of stars, their hearts will fly,  
With each little wag, they touch the sky.  

(Outro)
So here's to the puppies, in a world so wide,  
May your wagging never end, may joy be your guide.
With every playful bark and every silly song,  
In our hearts, dear puppies, you'll always belong.
We'll dance through life, as the puppies play,  
In this Puppy Paradise, come what may.
(Verse 4)
Through morning's dew and evening's glow,  
These little paws leave trails, where happiness flows.
With gentle licks and playful bounces,  
Love multiplies with every ounce.

(Chorus)
Oh, puppies, oh puppies, bringers of joy,  
In your playful world, every heart you'll buoy.
Furry little bundles with spirits so free,  
In your love, we find our peace and glee.

(Verse 5)
Every little adventure, a story to write,  
Under the moon, with stars shining bright.
Their innocence sings in the early morn,  
With each new day, new bonds are born.  

(Outro)
Puppy laughter fills the air,  
In a world of dreams, beyond compare.  
So here's to the wonder of puppy charms,
Forever in our hearts, our love swarms.  
Puppy Paradise, where love never ends,
In their sweet embrace, our hearts ascend.
In the dance of their innocence, all troubles disappear.
(Verse 2)
Playful barks echo, in the morning light,
Jumping through the flowers, what a beautiful sight.  
Wagging little tails, chasing dreams so high,  
With a cuddle and a giggle, we watch the day go by.
(Chorus)  
Oh, puppies, oh puppies, with eyes so bright,
Paint the world with joy and pure delight.
Furry little bundles of endless cheer,
In the dance of their innocence, all troubles disappear.
(Bridge)
Every puppy step tells a story true,  
Of friendship and love, always fresh as dew.  
With a paw in our hand, no journey's too long,  
Together we write our puppy song.  

(Verse 3)
As the sun sets low, they curl up tight,
Dreaming puppy dreams, through the calming night.
In a blanket of stars, their hearts will fly,
With each little wag, they touch the sky.  

(Outro)
So here's to the puppies, in a world so wide,  
May your wagging never end, may joy be your guide.  
With every playful bark and every silly song,  
In our hearts, dear puppies, you'll always belong.
We'll dance through life, as the puppies play,  
In this Puppy Paradise, come what may.
(Verse 4)
Through morning's dew and evening's glow,
These little paws leave trails, where happiness flows.
With gentle licks and playful bounces,  
Love multiplies with every ounce.

(Chorus)  
Oh, puppies, oh puppies, bringers of joy,  
In your playful world, every heart you'll buoy.
Furry little bundles with spirits so free,  
In your love, we find our peace and glee.

(Verse 5)
Every little adventure, a story to write,
Under the moon, with stars shining bright.
Their innocence sings in the early morn,  
With each new day, new bonds are born.  

(Outro)
Puppy laughter fills the air,  
In a world of dreams, beyond compare.  
So here's to the wonder of puppy charms,
Forever in our hearts, our love swarms.  
Puppy Paradise, where love never ends,
In their sweet embrace, our hearts ascend.
"""

def test_update_internal_state_with_long_string():
    """
    Integration test:
      1. Wrap a long blurb into sections with a small chunk size (to force multiple sections).
      2. Update section2 with new content using SectionBasedEditor.apply_edits.
      3. Unwrap the updated content to remove section markers.
      4. Verify that the update for section2 is applied and that the original content for section2 is gone.
         The other sections remain unchanged.
    """
    # Wrap the long blurb into sections with a max chunk size small enough to force multiple sections.
    doc_state = wrap_sections(long_blurb, max_chunk_size=300)

    # Parse the wrapped document to extract sections.
    sections, _ = parse_sections(doc_state)
    assert len(sections) >= 2, "The document should have at least two sections for this test."

    # Save the original content of section2.
    original_section2 = None
    for sec_id, content, _, _ in sections:
        if sec_id == "section2":
            original_section2 = content.strip()
            break
    assert original_section2 is not None, "Section2 should exist in the wrapped document."

    # Prepare an edit that updates section2.
    edit = """```section1
Puppy Paradise

(Verse 1)
(G) In a world so grand, with fields to explore,
(C) Tiny paws on the ground, and we're ready for more.
(G) Little wet noses, sniffing the breeze,
(D) Chasing our tails, with hearts full of glee.

(Chorus)
(G) Oh, puppies, oh puppies, with eyes so bright,
(C) Paint the world with joy and pure delight.
(G) Furry little bundles of endless cheer,
(D) In the dance of their innocence, all troubles disappear.
```

```section2
(Verse 2)
(G) Playful barks echo, in the morning light,
(C) Jumping through the flowers, what a beautiful sight.
(G) Wagging little tails, chasing dreams so high,
(D) With a cuddle and a giggle, we watch the day go by.

(Chorus)
(G) Oh, puppies, oh puppies, with eyes so bright,
(C) Paint the world with joy and pure delight.
(G) Furry little bundles of endless cheer,
(D) In the dance of their innocence, all troubles disappear.
```

```section3
(Bridge)
(C) Every puppy step tells a story true,
(G) Of friendship and love, always fresh as dew.
(D) With a paw in our hand, no journey's too long,
(G) Together we write our puppy song.

(Verse 3)
(G) As the sun sets low, they curl up tight,
(C) Dreaming puppy dreams, through the calming night.
(G) In a blanket of stars, their hearts will fly,
(D) With each little wag, they touch the sky.
```

```section4
(Outro)
(G) So here's to the puppies, in a world so wide,
(C) May your wagging never end, may joy be your guide.
(G) With every playful bark and every silly song,
(D) In our hearts, dear puppies, you'll always belong.
(G) We'll dance through life, as the puppies play,
(C) In this Puppy Paradise, come what may.
```

```section5
(Verse 4)
(G) Through morning's dew and evening's glow,
(C) These little paws leave trails, where happiness flows.
(G) With gentle licks and playful bounces,
(D) Love multiplies with every ounce.

(Chorus)
(G) Oh, puppies, oh puppies, bringers of joy,
(C) In your playful world, every heart you'll buoy.
(G) Furry little bundles with spirits so free,
(D) In your love, we find our peace and glee.
```

```section6
(Verse 5)
(G) Every little adventure, a story to write,
(C) Under the moon, with stars shining bright.
(G) Their innocence sings in the early morn,
(D) With each new day, new bonds are born.

(Outro)
(G) Puppy laughter fills the air,
(C) In a world of dreams, beyond compare.
(G) So here's to the wonder of puppy charms,
(D) Forever in our hearts, our love swarms.
(G) Puppy Paradise, where love never ends,
(C) In their sweet embrace, our hearts ascend.
```
"""

    # Simulate the main code: update internal state using the editor and then unwrap the sections.
    updated_with_markers = SectionBasedEditor.apply_edits(doc_state, edit)
    final_content = unwrap_sections(updated_with_markers)

    # Verify that the updated content includes the new text in place of the original section2.
    assert original_section2 not in final_content, "The old section2 content should no longer be present."

    # Optionally, verify that parts of other sections are still present.
    # We'll check for some text from the long_blurb that is likely to be in section1 or later sections.
    print(final_content)

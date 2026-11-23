using NUnit.Framework;

namespace Doji.AI.Transformers.Editor.Tests {

    public class TokenizerConfigTest {

        [Test]
        public void Deserialize_ReadsTokenObjectsAndStrings() {
            // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/tokenizer_config.json
            string json = @"{""add_prefix_space"":false,""bos_token"":{""__type"":""AddedToken"",""content"":""<|startoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false},""do_lower_case"":true,""eos_token"":{""__type"":""AddedToken"",""content"":""<|endoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false},""errors"":""replace"",""model_max_length"":77,""name_or_path"":""openai/clip-vit-large-patch14"",""pad_token"":""<|endoftext|>"",""special_tokens_map_file"":""./special_tokens_map.json"",""tokenizer_class"":""CLIPTokenizer"",""unk_token"":{""__type"":""AddedToken"",""content"":""<|endoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false}}";
            TokenizerConfig config = TokenizerConfig.Deserialize(json);

            Assert.That(config.BosToken.Content, Is.EqualTo("<|startoftext|>"));
            Assert.That(config.BosToken.Normalized, Is.True);
            Assert.That(config.BosToken.Special, Is.True, "A named special token is special.");
            Assert.That(config.PadToken.Content, Is.EqualTo("<|endoftext|>"));
            Assert.That(config.PadToken.Normalized, Is.False, "A special token from a string is not normalized.");
            Assert.That(config.ModelMaxLength, Is.EqualTo(77));
            Assert.That(config.TokenizerClass, Is.EqualTo("CLIPTokenizer"));
            Assert.That(config.SepToken, Is.Null);
        }

        [Test]
        public void Deserialize_ReadsAddedTokensAndHugeMaxLength() {
            string json = @"{""added_tokens_decoder"":{""49406"":{""content"":""<|startoftext|>"",""lstrip"":false,""normalized"":true,""rstrip"":false,""single_word"":false,""special"":true}},""model_max_length"":1000000000000000019884624838656,""padding_side"":""left"",""extra_special_tokens"":[""<a>"",""<b>""]}";
            TokenizerConfig config = TokenizerConfig.Deserialize(json);

            Assert.That(config.AddedTokensDecoder[49406].Content, Is.EqualTo("<|startoftext|>"));
            Assert.That(config.AddedTokensDecoder[49406].Special, Is.True);
            Assert.That(config.ModelMaxLength, Is.EqualTo(int.MaxValue));
            Assert.That(config.PaddingSide, Is.EqualTo(Side.Left));
            Assert.That(config.ExtraSpecialTokens.ConvertAll(t => t.Content), Is.EqualTo(new[] { "<a>", "<b>" }));
        }
    }
}

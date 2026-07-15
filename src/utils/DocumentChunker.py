class DocumentChunker:
    def __init__(
        self, 
        tokenizer, 
        chunk_size=254, 
        overlap=30, 
        min_tokens=35, 
        max_prefix_tokens=64
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tokens = min_tokens
        self.max_prefix_tokens = max_prefix_tokens

    def chunk_document(self, title_str, sections):
        """
        Takes a title and a list of section dictionaries (with 'heading' and 'text').
        Returns a list of chunked passage strings.
        """
        if not sections:
            return []
            
        doc_text = ""
        section_boundaries = [] # (start_char_idx, end_char_idx, heading_string)
        
        # 1. Reconstruct the full document text while tracking where sections start/end
        for sec in sections:
            heading = sec.get('heading', '').strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            body = sec.get('text', '').strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
            
            sec_str = ""
            if heading:
                suffix = "" if heading[-1] in ".!?:;" else "."
                sec_str += f"Section: {heading}{suffix} "
                
            if body:
                suffix = "" if body[-1] in ".!?" else "."
                sec_str += f"{body}{suffix} "
            
            start_c = len(doc_text)
            doc_text += sec_str
            end_c = len(doc_text)
            
            section_boundaries.append((start_c, end_c, heading))
            
        text_encoding = self.tokenizer(
            doc_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_offsets_mapping=True
        )
        text_tokens = text_encoding.input_ids
        text_offsets = text_encoding.offset_mapping
        
        passages = []
        start = 0
        
        while start < len(text_tokens):
            # 3. Find the valid character start offset
            start_char = None
            for offset in text_offsets[start:]:
                if offset is not None:
                    start_char = offset[0]
                    break
                    
            if start_char is None:
                break
                
            # 4. Determine active section
            active_heading = ""
            for (sec_start, sec_end, h) in section_boundaries:
                if sec_start <= start_char < sec_end:
                    active_heading = h
                    break
                    
            # 5. Construct the dynamic prefix
            if title_str and active_heading:
                suffix = "" if active_heading[-1] in ".!?" else "."
                prefix_str = f"{title_str} - Section: {active_heading}{suffix} "
            elif title_str:
                suffix = "" if title_str[-1] in ".!?" else "."
                prefix_str = f"{title_str}{suffix} "
            elif active_heading:
                suffix = "" if active_heading[-1] in ".!?" else "."
                prefix_str = f"Section: {active_heading}{suffix} "
            else:
                prefix_str = ""
                
            # Tokenize prefix to find its exact length
            prefix_encoding = self.tokenizer(
                prefix_str,
                add_special_tokens=False,
                max_length=self.max_prefix_tokens,
                truncation=True
            )
            prefix_tokens = prefix_encoding.input_ids
            
            if len(prefix_tokens) == self.max_prefix_tokens:
                prefix_str = self.tokenizer.decode(prefix_tokens).strip() + " "
                
            prefix_len = len(prefix_tokens)
            available_space = self.chunk_size - prefix_len
            
            # 6. Extract chunk body
            end = start + available_space
            chunk_text_ids = text_tokens[start:end]
            
            if len(chunk_text_ids) < self.min_tokens and start > 0:
                break
                
            chunk_offsets = [o for o in text_offsets[start:end] if o is not None]
            if not chunk_offsets:
                start += available_space
                continue
                
            chunk_start_char = chunk_offsets[0][0]
            chunk_end_char = chunk_offsets[-1][1]
            passage_text_body = doc_text[chunk_start_char:chunk_end_char]
            
            # 7. Deduplication Check
            clean_body = passage_text_body.lstrip()
            expected_heading_text = f"Section: {active_heading}"
            
            if active_heading and clean_body.startswith(expected_heading_text):
                clean_body = clean_body[len(expected_heading_text):].lstrip(" .:")
                
            # 8. Assemble the final passage
            final_passage_text = f"{prefix_str}{clean_body}".strip()
            passages.append(final_passage_text)
            
            if end >= len(text_tokens):
                break
                
            prev_start = start
            start = end - self.overlap
            
            if start <= prev_start:
                start = prev_start + 1
                
        return passages